//! Accuracy of the smallest singular value: TSQR against the normal
//! equations.

use nalgebra::{Matrix6, Vector6};
use rigidity_core::linalg::{reduce, singular_values};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next().max(f64::MIN_POSITIVE);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * self.next()).cos()
    }
}

/// Rows of J = Q·Σ·Vᵀ with prescribed singular values.
fn build(rows: usize, sigma: &[f64; 6], seed: u64) -> Vec<[f64; 6]> {
    let mut rng = Rng(seed);
    let mut columns: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..rows).map(|_| rng.normal()).collect())
        .collect();
    for _ in 0..2 {
        for j in 0..6 {
            let (left, right) = columns.split_at_mut(j);
            for column in left.iter().take(j) {
                let dot: f64 = right[0].iter().zip(column).map(|(a, b)| a * b).sum();
                for (value, base) in right[0].iter_mut().zip(column) {
                    *value -= dot * base;
                }
            }
            let norm: f64 = right[0].iter().map(|v| v * v).sum::<f64>().sqrt();
            for value in right[0].iter_mut() {
                *value /= norm;
            }
        }
    }
    let mut raw = Matrix6::zeros();
    for i in 0..6 {
        for j in 0..6 {
            raw[(i, j)] = rng.normal();
        }
    }
    let v = raw.qr().q();
    (0..rows)
        .map(|i| {
            let scaled = Vector6::from_iterator((0..6).map(|j| columns[j][i] * sigma[j]));
            let row = v * scaled;
            [row[0], row[1], row[2], row[3], row[4], row[5]]
        })
        .collect()
}

/// Superscript rendering of an exponent.
fn superscript(power: i32) -> String {
    let digits = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let mut text = String::new();
    if power < 0 {
        text.push('⁻');
    }
    for character in power.abs().to_string().chars() {
        text.push(digits[character.to_digit(10).unwrap() as usize]);
    }
    text
}

/// Draws the chart on logarithmic axes.
///
/// One axis only: both curves measure the same quantity, the relative
/// error of `σ_min`, so a second scale would be a sleight of hand.
fn write_svg(points: &[(f64, f64, f64)], path: &str) {
    const WIDTH: f64 = 800.0;
    const HEIGHT: f64 = 500.0;
    const LEFT: f64 = 82.0;
    const RIGHT: f64 = 178.0;
    const TOP: f64 = 92.0;
    const BOTTOM: f64 = 92.0;
    const X_MIN: f64 = 1.0;
    const X_MAX: f64 = 15.0;
    const Y_MIN: f64 = -16.0;
    const Y_MAX: f64 = 7.0;

    let plot_width = WIDTH - LEFT - RIGHT;
    let plot_height = HEIGHT - TOP - BOTTOM;
    let x_of = |kappa: f64| LEFT + (kappa.log10() - X_MIN) / (X_MAX - X_MIN) * plot_width;
    let y_of = |error: f64| TOP + (Y_MAX - error.log10()) / (Y_MAX - Y_MIN) * plot_height;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {WIDTH} {HEIGHT}\" \
width=\"{WIDTH}\" height=\"{HEIGHT}\" font-family=\"-apple-system, BlinkMacSystemFont, \
'Segoe UI', Roboto, sans-serif\">\n"
    ));

    // Roles rather than raw colours: dark mode substitutes them in one
    // place.
    svg.push_str(
        "<style>\n\
 :root { --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#8a8880;\n\
         --grid:#e6e5e0; --axis:#c9c7c0; --direct:#2a78d6; --normal:#eb6834; }\n\
 @media (prefers-color-scheme: dark) {\n\
 :root { --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8a8880;\n\
         --grid:#2e2e2b; --axis:#45443f; --direct:#3987e5; --normal:#d95926; }\n\
 }\n\
 .surface{fill:var(--surface)} .title{fill:var(--ink);font-size:16px;font-weight:600}\n\
 .subtitle{fill:var(--ink-2);font-size:12.5px} .tick{fill:var(--ink-2);font-size:11.5px}\n\
 .axis-title{fill:var(--ink-2);font-size:12px} .note{fill:var(--ink-3);font-size:11px}\n\
 .grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}\n\
 .series{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}\n\
 .marker{stroke:var(--surface);stroke-width:2}\n\
 .label{fill:var(--ink);font-size:12px;font-weight:600}\n\
 .legend{fill:var(--ink-2);font-size:12px}\n\
 .guide{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:4 4;opacity:.75}\n\
</style>\n",
    );
    svg.push_str(&format!(
        "<rect class=\"surface\" x=\"0\" y=\"0\" width=\"{WIDTH}\" height=\"{HEIGHT}\"/>\n"
    ));

    svg.push_str(&format!(
        "<text class=\"title\" x=\"{LEFT}\" y=\"28\">Relative error of σ<tspan font-size=\"11\" dy=\"3\">min</tspan></text>\n\
<text class=\"subtitle\" x=\"{LEFT}\" y=\"48\">Direct path through J against the normal \
equations JᵀJ · f64 · 20,000 rows</text>\n"
    ));

    // A legend is required for two or more series, even when the line
    // ends carry direct labels.
    for (index, (colour, label)) in [("--direct", "through J"), ("--normal", "through JᵀJ")]
        .into_iter()
        .enumerate()
    {
        let x = LEFT + index as f64 * 118.0;
        svg.push_str(&format!(
            "<circle cx=\"{x:.1}\" cy=\"70\" r=\"4\" fill=\"var({colour})\"/>\n\
<text class=\"legend\" x=\"{:.1}\" y=\"74\">{label}</text>\n",
            x + 11.0
        ));
    }

    // Gridlines at decades.
    for power in (2..=14).step_by(2) {
        let x = x_of(10f64.powi(power));
        svg.push_str(&format!(
            "<line class=\"grid\" x1=\"{x:.1}\" y1=\"{TOP}\" x2=\"{x:.1}\" \
y2=\"{:.1}\"/>\n<text class=\"tick\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">10{}</text>\n",
            TOP + plot_height,
            TOP + plot_height + 20.0,
            superscript(power)
        ));
    }
    for power in (-15..=6).step_by(3) {
        let y = y_of(10f64.powi(power));
        svg.push_str(&format!(
            "<line class=\"grid\" x1=\"{LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" \
y2=\"{y:.1}\"/>\n<text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">10{}</text>\n",
            LEFT + plot_width,
            LEFT - 10.0,
            y + 4.0,
            superscript(power)
        ));
    }

    // Guides: the project's operating point and the level of total loss.
    let floor_x = x_of(1.7e7);
    svg.push_str(&format!(
        "<line class=\"guide\" x1=\"{floor_x:.1}\" y1=\"{TOP}\" x2=\"{floor_x:.1}\" \
y2=\"{:.1}\"/>\n<text class=\"note\" x=\"{:.1}\" y=\"{:.1}\">f32 storage ceiling</text>\n",
        TOP + plot_height,
        floor_x + 6.0,
        TOP + 14.0
    ));
    let full_loss = y_of(1.0);
    svg.push_str(&format!(
        "<line class=\"guide\" x1=\"{LEFT}\" y1=\"{full_loss:.1}\" x2=\"{:.1}\" \
y2=\"{full_loss:.1}\"/>\n<text class=\"note\" x=\"{LEFT}\" y=\"{:.1}\">100 % error: the value is entirely lost</text>\n",
        LEFT + plot_width,
        full_loss - 6.0
    ));

    svg.push_str(&format!(
        "<line class=\"axis\" x1=\"{LEFT}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"/>\n\
<line class=\"axis\" x1=\"{LEFT}\" y1=\"{TOP}\" x2=\"{LEFT}\" y2=\"{:.1}\"/>\n",
        TOP + plot_height,
        LEFT + plot_width,
        TOP + plot_height,
        TOP + plot_height
    ));

    for (index, (colour, label)) in [("--direct", "through J"), ("--normal", "through JᵀJ")]
        .into_iter()
        .enumerate()
    {
        let value_of = |p: &(f64, f64, f64)| if index == 0 { p.1 } else { p.2 };
        let path: Vec<String> = points
            .iter()
            .map(|p| format!("{:.1},{:.1}", x_of(p.0), y_of(value_of(p).max(1e-16))))
            .collect();
        svg.push_str(&format!(
            "<polyline class=\"series\" stroke=\"var({colour})\" points=\"{}\"/>\n",
            path.join(" ")
        ));
        for p in points {
            svg.push_str(&format!(
                "<circle class=\"marker\" cx=\"{:.1}\" cy=\"{:.1}\" r=\"4\" \
fill=\"var({colour})\"/>\n",
                x_of(p.0),
                y_of(value_of(p).max(1e-16))
            ));
        }
        let last = points.last().unwrap();
        svg.push_str(&format!(
            "<text class=\"label\" x=\"{:.1}\" y=\"{:.1}\">{label}</text>\n",
            x_of(last.0) + 14.0,
            y_of(value_of(last).max(1e-16)) + 4.0
        ));
    }

    svg.push_str(&format!(
        "<text class=\"axis-title\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">\
condition number κ(J)</text>\n",
        LEFT + plot_width / 2.0,
        HEIGHT - 46.0
    ));
    svg.push_str(&format!(
        "<text class=\"note\" x=\"{LEFT}\" y=\"{:.1}\">Slope 1 against slope 2: the \
error grows as ε·κ and as ε·κ². Below 10⁻¹⁵ both curves hit machine precision.</text>\n",
        HEIGHT - 22.0
    ));
    svg.push_str("</svg>\n");
    std::fs::write(path, svg).unwrap();
}

fn main() {
    println!("{:>10}  {:>12}  {:>12}", "kappa", "TSQR", "normal eq.");
    let mut csv = String::from("kappa,tsqr,normal_equations\n");
    let mut measured: Vec<(f64, f64, f64)> = Vec::new();
    for power in [4i32, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48] {
        let smallest = 2f64.powi(-power);
        let sigma = [1.0, 0.5, 0.25, 0.125, 0.0625, smallest];
        let kappa = 1.0 / smallest;
        let rows = build(20_000, &sigma, 0xC0FFEE + power as u64);

        let triangle = reduce::<6, _>(rows.len(), |i| Some(rows[i]));
        let from_tsqr = singular_values(triangle.triangle())[5];

        let mut hessian = [[0.0f64; 6]; 6];
        for row in &rows {
            for i in 0..6 {
                for j in 0..6 {
                    hessian[i][j] += row[i] * row[j];
                }
            }
        }
        let from_normal = singular_values(&hessian)[5].sqrt();

        let err_tsqr = (from_tsqr - smallest).abs() / smallest;
        let err_normal = (from_normal - smallest).abs() / smallest;
        println!("{kappa:>10.2e}  {err_tsqr:>12.3e}  {err_normal:>12.3e}");
        csv.push_str(&format!("{kappa:e},{err_tsqr:e},{err_normal:e}\n"));
        measured.push((kappa, err_tsqr, err_normal));
    }
    std::fs::write("figures/sigma-min-accuracy.csv", csv).unwrap();
    write_svg(&measured, "figures/sigma-min-accuracy.svg");
    println!("\nwritten: figures/sigma-min-accuracy.svg and .csv");
}
