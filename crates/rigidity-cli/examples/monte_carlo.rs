//! Predicted spread against the empirical one: the repository's second
//! figure.

use rigidity_cli::monte_carlo::{SceneOutcome, TrialConfig, run_all};
use rigidity_core::observability::Observability;

/// One point on the chart.
struct Point {
    predicted: f64,
    empirical: f64,
    reliable: bool,
}

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

fn write_svg(points: &[Point], excluded: usize, excluded_error: f64, path: &str) {
    const WIDTH: f64 = 800.0;
    const HEIGHT: f64 = 540.0;
    const LEFT: f64 = 86.0;
    const RIGHT: f64 = 72.0;
    const TOP: f64 = 104.0;
    const BOTTOM: f64 = 130.0;
    const MIN: f64 = -5.5;
    const MAX: f64 = 4.0;

    let span = WIDTH - LEFT - RIGHT;
    let height = HEIGHT - TOP - BOTTOM;
    let x_of = |value: f64| LEFT + (value.log10() - MIN) / (MAX - MIN) * span;
    let y_of = |value: f64| TOP + (MAX - value.log10()) / (MAX - MIN) * height;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {WIDTH} {HEIGHT}\" \
width=\"{WIDTH}\" height=\"{HEIGHT}\" font-family=\"-apple-system, BlinkMacSystemFont, \
'Segoe UI', Roboto, sans-serif\">\n"
    ));
    svg.push_str(
        "<style>\n\
 :root { --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#8a8880;\n\
         --grid:#e6e5e0; --axis:#c9c7c0; --reliable:#2a78d6; --flagged:#eb6834; }\n\
 @media (prefers-color-scheme: dark) {\n\
 :root { --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8a8880;\n\
         --grid:#2e2e2b; --axis:#45443f; --reliable:#3987e5; --flagged:#d95926; }\n\
 }\n\
 .surface{fill:var(--surface)} .title{fill:var(--ink);font-size:16px;font-weight:600}\n\
 .subtitle{fill:var(--ink-2);font-size:12.5px} .tick{fill:var(--ink-2);font-size:11.5px}\n\
 .axis-title{fill:var(--ink-2);font-size:12px} .note{fill:var(--ink-3);font-size:11px}\n\
 .legend{fill:var(--ink-2);font-size:12px}\n\
 .grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}\n\
 .marker{stroke:var(--surface);stroke-width:2}\n\
 .guide{stroke:var(--ink-3);stroke-width:1.5;stroke-dasharray:5 4;opacity:.8;fill:none}\n\
</style>\n",
    );
    svg.push_str(&format!(
        "<rect class=\"surface\" x=\"0\" y=\"0\" width=\"{WIDTH}\" height=\"{HEIGHT}\"/>\n"
    ));
    svg.push_str(&format!(
        "<text class=\"title\" x=\"{LEFT}\" y=\"30\">Predicted spread against \
empirical</text>\n<text class=\"subtitle\" x=\"{LEFT}\" y=\"50\">1000 trials per scene · \
7 scenes · analytical and estimated normals · noise σ 1 mm</text>\n"
    ));

    for (index, (colour, label)) in [
        ("--reliable", "marked reliable (HIGH/MEDIUM)"),
        ("--flagged", "marked unreliable (LOW)"),
    ]
    .into_iter()
    .enumerate()
    {
        let x = LEFT + index as f64 * 250.0;
        svg.push_str(&format!(
            "<circle cx=\"{x:.1}\" cy=\"74\" r=\"4\" fill=\"var({colour})\"/>\n\
<text class=\"legend\" x=\"{:.1}\" y=\"78\">{label}</text>\n",
            x + 11.0
        ));
    }

    for power in -5..=4i32 {
        if power > 4 {
            continue;
        }
        let x = x_of(10f64.powi(power));
        let y = y_of(10f64.powi(power));
        svg.push_str(&format!(
            "<line class=\"grid\" x1=\"{x:.1}\" y1=\"{TOP}\" x2=\"{x:.1}\" y2=\"{:.1}\"/>\n\
<text class=\"tick\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">10{}</text>\n\
<line class=\"grid\" x1=\"{LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/>\n\
<text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">10{}</text>\n",
            TOP + height,
            TOP + height + 20.0,
            superscript(power),
            LEFT + span,
            LEFT - 10.0,
            y + 4.0,
            superscript(power)
        ));
    }

    // The line of perfect agreement.
    svg.push_str(&format!(
        "<line class=\"guide\" x1=\"{LEFT}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{TOP}\"/>\n\
<text class=\"note\" x=\"{:.1}\" y=\"{:.1}\" transform=\"rotate(-45 {:.1} {:.1})\">\
prediction = empirical</text>\n",
        TOP + height,
        LEFT + span,
        LEFT + span * 0.52,
        TOP + height * 0.44,
        LEFT + span * 0.52,
        TOP + height * 0.44
    ));
    svg.push_str(&format!(
        "<line class=\"axis\" x1=\"{LEFT}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"/>\n\
<line class=\"axis\" x1=\"{LEFT}\" y1=\"{TOP}\" x2=\"{LEFT}\" y2=\"{:.1}\"/>\n",
        TOP + height,
        LEFT + span,
        TOP + height,
        TOP + height
    ));

    for point in points {
        let colour = if point.reliable {
            "--reliable"
        } else {
            "--flagged"
        };
        svg.push_str(&format!(
            "<circle class=\"marker\" cx=\"{:.1}\" cy=\"{:.1}\" r=\"5\" fill=\"var({colour})\"/>\n",
            x_of(point.predicted),
            y_of(point.empirical)
        ));
    }

    svg.push_str(&format!(
        "<text class=\"axis-title\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">\
predicted spread σ_noise/σ'ᵢ, metres</text>\n",
        LEFT + span / 2.0,
        HEIGHT - 64.0
    ));
    svg.push_str(&format!(
        "<text class=\"axis-title\" transform=\"rotate(-90 18 {:.1})\" x=\"18\" y=\"{:.1}\" \
text-anchor=\"middle\">empirical spread, metres</text>\n",
        TOP + height / 2.0,
        TOP + height / 2.0
    ));
    svg.push_str(&format!(
        "<text class=\"note\" x=\"{LEFT}\" y=\"{:.1}\">{excluded} directions with an \
infinite prediction are not shown: ICP does not move the pose along them,</text>\n\
<text class=\"note\" x=\"{LEFT}\" y=\"{:.1}\">so the error stays at the initial \
perturbation, about {excluded_error:.1e} m. The four points on the right sit at the \
f32 ceiling.</text>\n",
        HEIGHT - 40.0,
        HEIGHT - 24.0
    ));
    svg.push_str("</svg>\n");
    std::fs::write(path, svg).unwrap();
}

fn print_table(outcomes: &[SceneOutcome]) {
    println!(
        "{:<18} {:>3} {:>11} {:>11} {:>8} {:>9}  class",
        "scene", "i", "predicted", "empirical", "ratio", "bias"
    );
    for outcome in outcomes {
        for direction in &outcome.directions {
            println!(
                "{:<18} {:>3} {:>11.3e} {:>11.3e} {:>8.2} {:>9.1e}  {}",
                if direction.index == 0 {
                    outcome.kind.name()
                } else {
                    ""
                },
                direction.index,
                direction.predicted,
                direction.empirical,
                direction.ratio(),
                direction.bias,
                direction.observability.label()
            );
        }
    }
}

fn main() {
    let trials: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000);
    let points: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(800);

    let mut plot: Vec<Point> = Vec::new();
    let mut csv = String::from("scene,normals,direction,predicted,empirical,ratio,bias,class\n");
    let mut excluded = 0usize;
    let mut excluded_error = 0.0f64;
    let mut reliable_ratios: Vec<f64> = Vec::new();

    for estimated in [false, true] {
        let config = TrialConfig {
            trials,
            points_per_face: points,
            estimated_normals: estimated,
            ..TrialConfig::default()
        };
        println!(
            "\n=== normals: {} ===",
            if estimated { "estimated" } else { "analytical" }
        );
        let outcomes = run_all(&config);
        print_table(&outcomes);

        for outcome in &outcomes {
            for direction in &outcome.directions {
                let reliable = direction.observability != Observability::Low;
                csv.push_str(&format!(
                    "{},{},{},{:e},{:e},{:e},{:e},{}\n",
                    outcome.kind.name(),
                    if estimated { "estimated" } else { "analytical" },
                    direction.index,
                    direction.predicted,
                    direction.empirical,
                    direction.ratio(),
                    direction.bias,
                    direction.observability.label()
                ));
                if direction.predicted.is_finite() {
                    plot.push(Point {
                        predicted: direction.predicted,
                        empirical: direction.empirical,
                        reliable,
                    });
                    if reliable {
                        reliable_ratios.push(direction.ratio());
                    }
                } else {
                    excluded += 1;
                    excluded_error = excluded_error.max(direction.empirical);
                }
            }
        }
    }

    reliable_ratios.sort_by(f64::total_cmp);
    let median = reliable_ratios[reliable_ratios.len() / 2];
    println!(
        "\nreliable directions: {}, empirical/predicted ratio: \
median {:.3}, range {:.3}…{:.3}",
        reliable_ratios.len(),
        median,
        reliable_ratios[0],
        reliable_ratios[reliable_ratios.len() - 1]
    );

    std::fs::write("figures/monte-carlo.csv", csv).unwrap();
    write_svg(&plot, excluded, excluded_error, "figures/monte-carlo.svg");
    println!("written: figures/monte-carlo.svg and .csv");
}
