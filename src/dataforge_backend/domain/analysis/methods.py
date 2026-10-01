"""The analysis methods a member can choose between, and the ways of working (modes) for each."""

from typing import Any

ANALYSIS_METHODS: list[dict[str, Any]] = [
    {
        "id": "summarise",
        "name": "Summarise a measure",
        "question": "What is typical, and how much does it vary?",
        "description": "Describe one measure across groups or months: its typical value, how spread out it is, "
        "and which values are unusual.",
        "techniques": [
            {
                "name": "Median or mean",
                "plain": "The median is the middle value and ignores extremes. The mean is the arithmetic average "
                "and is pulled by them.",
            },
            {
                "name": "Spread",
                "plain": "Standard deviation and the interquartile range (the middle half of values) show how far "
                "values sit from typical.",
            },
            {
                "name": "Distribution",
                "plain": "A histogram shows whether values cluster, spread evenly or lean to one side.",
            },
            {
                "name": "Unusual values",
                "plain": "Tukey fences flag values more than 1.5 × the interquartile range beyond the middle half. "
                "You decide whether to keep them.",
            },
        ],
        "cautions": [
            "Money is usually skewed. Prefer the median for a typical fee or revenue.",
            "A summary over a handful of units is fragile. Check how many units it covers.",
        ],
        "examples": ["How many active learners are there this month?", "What is the average fee paid?"],
        "modes": ["build", "ask", "code"],
    },
    {
        "id": "compare",
        "name": "Compare groups",
        "question": "Which groups are higher or lower?",
        "description": "Put groups side by side on one measure, as totals, monthly averages, shares of the whole "
        "or an index against the average.",
        "techniques": [
            {
                "name": "Total or monthly average",
                "plain": "Totals favour groups that were active for longer. A monthly average or median puts them "
                "on the same footing.",
            },
            {"name": "Share of total", "plain": "Each group's part of the whole, for measures that add up."},
            {
                "name": "Index against the average",
                "plain": "100 is the average group. 120 means 20% above it.",
            },
            {
                "name": "Pooled or averaged rates",
                "plain": "A pooled rate divides the total numerator by the total denominator. Averaging group rates "
                "gives a small group the same weight as a large one.",
            },
        ],
        "cautions": [
            "Small groups swing more from month to month. Look at their size before ranking them.",
            "For rates, pooling is usually right. Averaging rates can mislead when groups differ in size.",
        ],
        "examples": [
            "What was the average fee by region last quarter?",
            "Which region has the most enrolments?",
        ],
        "modes": ["build", "ask", "code"],
    },
    {
        "id": "trend",
        "name": "Track a trend",
        "question": "How has it changed over time?",
        "description": "Follow a measure month by month as a level, a month-on-month change or a running total, "
        "with optional smoothing and one line per group.",
        "techniques": [
            {"name": "Level", "plain": "The value in each month."},
            {"name": "Month-on-month change", "plain": "The percentage change from the month before."},
            {"name": "Running total", "plain": "Everything so far, added up month by month."},
            {
                "name": "Three-month moving average",
                "plain": "Smooths out single-month noise so the direction is easier to see.",
            },
            {"name": "Split by group", "plain": "One line per group, to see whether they move together."},
        ],
        "cautions": [
            "Seasons matter. Compare like with like before calling a move a trend.",
            "A short series is fragile. Treat a single month's jump with care.",
        ],
        "examples": [
            "How has completion rate changed by course this year?",
            "How have enrolments moved since October?",
        ],
        "modes": ["build", "ask", "code"],
    },
    {
        "id": "relationship",
        "name": "Test a relationship",
        "question": "Do two measures move together?",
        "description": "Plot two measures against each other and measure how closely they move together.",
        "techniques": [
            {
                "name": "Pearson correlation (r)",
                "plain": "Measures a straight-line relationship from −1 to 1. Sensitive to extreme values.",
            },
            {
                "name": "Spearman correlation (ρ)",
                "plain": "Works on ranks, so it also catches curved relationships and resists extreme values.",
            },
            {
                "name": "Scatter plot with a fitted line",
                "plain": "Each dot is one unit. The line is the best straight-line fit.",
            },
            {
                "name": "Unit of analysis",
                "plain": "Choose what one dot is: a group in a month, a pair of groups, and so on.",
            },
        ],
        "cautions": [
            "Two measures moving together does not mean one causes the other.",
            "Each dot is a group, not a person. Do not read the result as true for each individual.",
            "With fewer than about ten dots, the correlation is unstable.",
        ],
        "examples": [],
        "modes": ["build", "code"],
        "unavailable": {
            "ask": "The assistant answers with shared metrics. Correlations are built here or written in SQL."
        },
    },
    {
        "id": "explain",
        "name": "Explain a change",
        "question": "What drove the change between two months?",
        "description": "Split the change in a total between two months into each group's contribution, so you can "
        "see where it came from.",
        "techniques": [
            {
                "name": "Contribution analysis",
                "plain": "Each group's change between the two months. The contributions add up to the total change.",
            },
            {"name": "Share of the change", "plain": "How much of the total change each group accounts for."},
            {
                "name": "Automatic alerts",
                "plain": "Monitors watch key metrics and explain their moves for you. Open Monitors to see them.",
            },
        ],
        "cautions": [
            "Contributions work for totals such as revenue or enrolments. For a rate, explain its numerator and "
            "denominator separately.",
            "A group can move against the total. Read the signs, not only the sizes.",
        ],
        "examples": [],
        "modes": ["build", "code"],
        "unavailable": {
            "ask": "Root causes are computed as contributions here, or explained on an alert in Monitors."
        },
    },
    {
        "id": "predict",
        "name": "Predict an outcome",
        "question": "How likely is an outcome, and why?",
        "description": "Score a cohort's likelihood of an outcome, with a ranked and documented breakdown of the "
        "factors behind the score.",
        "techniques": [
            {
                "name": "Documented weighted scoring",
                "plain": "A deterministic, published heuristic that a trained model can later replace.",
            },
            {
                "name": "Factor breakdown",
                "plain": "Each factor's weight and direction. The weights always add up to 1.",
            },
            {
                "name": "Recorded model version",
                "plain": "Every score records the version that produced it, for audit.",
            },
        ],
        "cautions": [
            "Scores describe cohorts, not individual people. Personal data stays masked.",
            "A score is a likelihood, not a certainty.",
        ],
        "examples": [],
        "modes": ["guided"],
        "unavailable": {
            "ask": "Predictions use the guided scorer so every factor is documented.",
            "build": "Predictions use the guided scorer so every factor is documented.",
            "code": "The scoring heuristic is fixed and versioned, so it is not edited here.",
        },
    },
]

ANALYSIS_MODES: dict[str, dict[str, str]] = {
    "ask": {
        "label": "Ask in plain English",
        "short": "Ask",
        "description": "Type a question. The assistant grounds it in shared metrics, runs it, and shows the SQL behind the answer.",
        "bestFor": "A quick answer without setting anything up.",
    },
    "build": {
        "label": "Build a chart",
        "short": "Build",
        "description": "Choose the fields and how the numbers are calculated, by point and click. The chart, table "
        "and SQL update as you go.",
        "bestFor": "Deciding exactly how the numbers are worked out.",
    },
    "code": {
        "label": "Write SQL",
        "short": "Code",
        "description": "Start from a template for the method, or your own query, and run it in the sandbox.",
        "bestFor": "Full control: joins, custom logic and notebooks.",
    },
    "guided": {
        "label": "Guided scoring",
        "short": "Predict",
        "description": "Choose an outcome and a cohort. You get a score and the factors behind it.",
        "bestFor": "Predictions with a documented explanation.",
    },
}

METHOD_NAMES: dict[str, str] = {method["id"]: method["name"] for method in ANALYSIS_METHODS}
