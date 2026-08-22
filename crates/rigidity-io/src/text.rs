//! Delimited text: `.txt` and `.csv`.
//!
//! The format everyone has and nobody specified. A scanner, a script or a
//! colleague hands over a file of numbers with one point per line, and the
//! only things that vary are the delimiter and whether the first line names
//! the columns or is already data.
//!
//! Both are decided by looking, and neither is guessed at more than once.
//! [`read_text`] takes the delimiter from the first line that holds data and
//! reads the whole file with it, rather than re-deciding per line — a file
//! whose separator changes half way through is a broken file, and reading it
//! anyway would turn a diagnosable error into silently wrong coordinates.
//!
//! [`write_text`] writes three numbers a line and nothing else: no header,
//! because a header is the part most likely to make another tool refuse the
//! file, and the reader here does not need one.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use nalgebra::Vector3;
use rigidity_core::PointCloud;

use crate::IoError;

/// How a file separates its columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Delimiter {
    /// Runs of spaces or tabs, the usual `.txt`.
    Whitespace,
    /// A single character: `,` or `;`.
    Character(char),
}

impl Delimiter {
    /// Splits a line, keeping empty fields when the delimiter is a
    /// character so that column positions survive a missing value.
    fn split(self, line: &str) -> Vec<&str> {
        match self {
            Self::Whitespace => line.split_whitespace().collect(),
            Self::Character(c) => line.split(c).map(str::trim).collect(),
        }
    }

    /// Guesses from a line, preferring an explicit separator to whitespace.
    ///
    /// A comma decides it even when spaces are also present, because
    /// `1.0, 2.0, 3.0` is comma-separated with padding and never three
    /// fields of whitespace with stray commas attached.
    fn of(line: &str) -> Self {
        for candidate in [',', ';'] {
            if line.contains(candidate) {
                return Self::Character(candidate);
            }
        }
        Self::Whitespace
    }
}

/// Whether a line is a comment or has nothing on it.
fn skippable(line: &str) -> bool {
    line.is_empty() || line.starts_with('#') || line.starts_with("//")
}

/// Finds the column whose name matches one of the candidates.
fn named(header: &[&str], names: &[&str]) -> Option<usize> {
    header.iter().position(|column| {
        let trimmed = column.trim().trim_matches('"').to_ascii_lowercase();
        names.iter().any(|candidate| trimmed == *candidate)
    })
}

/// Which fields hold the coordinates, and whether the first line is data.
///
/// A header is recognised by what it is not: if the first three columns of
/// the first line all parse as numbers, the line is data and there is no
/// header. That is a stronger test than looking for the letter `x`, because
/// plenty of headers call the column `X (m)` or `//X` and plenty of data
/// files start with a line that happens to contain letters in a later
/// column.
fn columns(first: &[&str]) -> ([usize; 3], bool) {
    let numeric = first
        .iter()
        .take(3)
        .filter(|field| field.trim().parse::<f64>().is_ok())
        .count();
    if numeric == 3 && first.len() >= 3 {
        return ([0, 1, 2], false);
    }
    let x = named(first, &["x", "x(m)", "x [m]", "//x"]);
    let y = named(first, &["y", "y(m)", "y [m]"]);
    let z = named(first, &["z", "z(m)", "z [m]"]);
    match (x, y, z) {
        (Some(x), Some(y), Some(z)) => ([x, y, z], true),
        // A header that does not name its coordinates still tells us it is
        // a header. Falling back to the first three columns is the same
        // assumption the headerless case makes, and it is better than
        // refusing a file whose columns are called `East North Up`.
        _ => ([0, 1, 2], true),
    }
}

/// Reads a cloud from a delimited text file.
///
/// The origin is placed at the centre of the bounding box: text files are
/// where global coordinates arrive, and `f32` storage cannot hold a UTM
/// easting directly. See `PointCloud::with_origin`.
pub fn read_text(path: &Path) -> Result<PointCloud, IoError> {
    let mut reader = BufReader::new(File::open(path)?);

    // The first line that is neither blank nor a comment decides both
    // questions, and is then re-read as data if it turns out to be data.
    let mut line = String::new();
    let first = loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(IoError::BadHeader("the file holds no data".into()));
        }
        if !skippable(line.trim()) {
            break line.trim().to_owned();
        }
    };

    let delimiter = Delimiter::of(&first);
    let fields = delimiter.split(&first);
    let ([x, y, z], had_header) = columns(&fields);
    let needed = x.max(y).max(z) + 1;

    let mut points: Vec<Vector3<f64>> = Vec::new();
    let mut minimum = Vector3::repeat(f64::INFINITY);
    let mut maximum = Vector3::repeat(f64::NEG_INFINITY);

    let mut take = |fields: &[&str]| -> Result<(), IoError> {
        if fields.len() < needed {
            return Ok(());
        }
        let parse = |index: usize| -> Result<f64, IoError> {
            fields[index]
                .trim()
                .parse()
                .map_err(|_| IoError::BadNumber(fields[index].to_string()))
        };
        let point = Vector3::new(parse(x)?, parse(y)?, parse(z)?);
        if point.iter().all(|value| value.is_finite()) {
            minimum = minimum.inf(&point);
            maximum = maximum.sup(&point);
            points.push(point);
        }
        Ok(())
    };

    if !had_header {
        take(&fields)?;
    }
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if skippable(trimmed) {
            continue;
        }
        take(&delimiter.split(trimmed))?;
    }

    if points.is_empty() {
        return Ok(PointCloud::new());
    }
    let origin = (minimum + maximum) * 0.5;
    let mut cloud = PointCloud::with_origin(origin);
    for point in &points {
        cloud.push(*point);
    }
    Ok(cloud)
}

/// Writes a cloud as one point a line.
///
/// The delimiter follows the extension — a space for `.txt`, a comma for
/// `.csv` — because that is the only thing either extension actually
/// promises anyone.
///
/// Coordinates are absolute and `f64`, formatted at the shortest decimal
/// that reads back as the identical value. Writing the stored `f32` offsets
/// would be smaller and would lose a quarter of a metre at four million
/// metres north, which is the mistake PCD made here once already and the
/// whole reason the core stores an `f64` origin.
pub fn write_text(cloud: &PointCloud, path: &Path) -> Result<(), IoError> {
    let comma = path
        .extension()
        .and_then(|end| end.to_str())
        .is_some_and(|end| end.eq_ignore_ascii_case("csv"));
    let separator = if comma { "," } else { " " };

    let mut out = BufWriter::new(File::create(path)?);
    for index in 0..cloud.len() {
        let point = cloud.point(index);
        writeln!(
            out,
            "{:?}{separator}{:?}{separator}{:?}",
            point.x, point.y, point.z
        )?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `body` to a uniquely named temporary file and reads it back.
    fn read(name: &str, body: &str) -> Result<PointCloud, IoError> {
        let path = std::env::temp_dir().join(format!("rigidity-io-text-{name}"));
        std::fs::write(&path, body).expect("a temporary file");
        let cloud = read_text(&path);
        std::fs::remove_file(&path).ok();
        cloud
    }

    fn points(cloud: &PointCloud) -> Vec<[f64; 3]> {
        (0..cloud.len())
            .map(|index| {
                let p = cloud.point(index);
                [p.x, p.y, p.z]
            })
            .collect()
    }

    const EXPECTED: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.5]];

    /// The same three points, said eight ways.
    ///
    /// Every one of these is a file somebody has actually handed over, and
    /// the round-trip test cannot reach any of them: it only ever reads
    /// back what this module's own writer produced, which is one of the
    /// eight.
    #[test]
    fn the_shapes_a_text_file_arrives_in() {
        let cases: [(&str, &str); 8] = [
            ("bare-space", "1 2 3\n4 5 6\n7 8 9.5\n"),
            ("bare-tab", "1\t2\t3\n4\t5\t6\n7\t8\t9.5\n"),
            ("ragged-space", "  1   2 3\n4 5   6\n 7 8 9.5  \n"),
            ("bare-comma", "1,2,3\n4,5,6\n7,8,9.5\n"),
            ("padded-comma", "1, 2, 3\n4, 5, 6\n7, 8, 9.5\n"),
            ("semicolon", "1;2;3\n4;5;6\n7;8;9.5\n"),
            ("named-header", "x,y,z\n1,2,3\n4,5,6\n7,8,9.5\n"),
            (
                "comments-and-blanks",
                "# station 4\n\n1 2 3\n\n// noise\n4 5 6\n7 8 9.5\n",
            ),
        ];
        for (name, body) in cases {
            let cloud = read(name, body).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(points(&cloud), EXPECTED, "{name}");
        }
    }

    /// Columns are found by name, not by position, when there is a header.
    #[test]
    fn a_header_puts_the_columns_where_it_says() {
        let cloud = read(
            "reordered",
            "id,z,intensity,x,y\n0,3,-1,1,2\n1,6,-1,4,5\n2,9.5,-1,7,8\n",
        )
        .expect("a reordered header should be read by name");
        assert_eq!(points(&cloud), EXPECTED);
    }

    /// Columns past the third are ignored rather than refused.
    #[test]
    fn extra_columns_are_not_an_obstacle() {
        let cloud = read("extra", "1 2 3 128 0.4\n4 5 6 130 0.5\n7 8 9.5 99 0.6\n")
            .expect("intensity and the rest are somebody else's business");
        assert_eq!(points(&cloud), EXPECTED);
    }

    /// A header whose columns are not called x, y and z is still a header.
    ///
    /// Falling through to the first three columns makes the same assumption
    /// the headerless case makes; refusing the file would be a reader that
    /// knows the answer and declines to give it.
    #[test]
    fn an_unnamed_header_is_recognised_as_one() {
        let cloud = read("east-north-up", "East North Up\n1 2 3\n4 5 6\n7 8 9.5\n")
            .expect("a header that does not name x should not lose its file");
        assert_eq!(points(&cloud), EXPECTED);
    }

    /// The extension picks the separator, and only the extension.
    #[test]
    fn csv_is_written_with_commas_and_txt_with_spaces() {
        let mut cloud = PointCloud::new();
        for point in EXPECTED {
            cloud.push(Vector3::new(point[0], point[1], point[2]));
        }
        for (extension, separator) in [("txt", " "), ("csv", ",")] {
            let path = std::env::temp_dir().join(format!("rigidity-io-sep.{extension}"));
            write_text(&cloud, &path).expect("the cloud should write");
            let text = std::fs::read_to_string(&path).expect("and be readable");
            assert_eq!(
                text.lines().next(),
                Some(format!("1.0{separator}2.0{separator}3.0").as_str()),
                "{extension}"
            );
            std::fs::remove_file(&path).ok();
        }
    }

    /// A file with nothing in it is an error, not an empty cloud.
    ///
    /// The difference matters: an empty cloud reads as "this scan saw
    /// nothing", and a scanner that saw nothing is a different problem from
    /// a file that was never written.
    #[test]
    fn an_empty_file_says_so() {
        for (name, body) in [("empty", ""), ("only-comments", "# nothing here\n\n")] {
            assert!(
                matches!(read(name, body), Err(IoError::BadHeader(_))),
                "{name} was accepted"
            );
        }
    }

    /// A number that is not one is reported rather than skipped.
    #[test]
    fn a_bad_number_is_an_error() {
        assert!(matches!(
            read("bad-number", "1 2 3\n4 five 6\n"),
            Err(IoError::BadNumber(_))
        ));
    }
}
