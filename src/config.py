"""Central configuration for tunable pipeline constants."""

# Tier routing: above this → Tier 1 (table cells), below → Tier 2/3
VISION_CONFIDENCE_THRESHOLD = 0.85

# Coordinate grounding: fuzzy match threshold for value → word_box matching
GROUNDING_THRESHOLD = 0.80

# Field label grounding: threshold for matching field labels to word_boxes
LABEL_GROUNDING_THRESHOLD = 0.70

# Tier 2 spatial clustering: Y-axis tolerance for grouping word_boxes into rows
Y_CLUSTERING_TOLERANCE = 0.005

# Tier 2 spatial clustering: X-gap threshold for detecting column boundaries
X_GAP_THRESHOLD = 0.02

# Maximum concurrent Gemini API calls
GEMINI_SEMAPHORE = 15
