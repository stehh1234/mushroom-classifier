# The mushroom data

`mushrooms.csv`: 8,124 mushrooms of 23 species of gilled mushrooms in the Agaricus and Lepiota
families, described from *The Audubon Society Field Guide to North American Mushrooms* (1981).
Source: [UCI Machine Learning Repository, Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)
(the same file is on Kaggle as `mushrooms.csv`).

- `class`: **the target**, `e` edible (4,208) or `p` poisonous (3,916). The field guide puts
  "unknown edibility, not recommended" in the poisonous class.
- 22 features, every one a single letter. `app/codes.py` translates them into words; the API
  only speaks words.
- `stalk-root` is `?` for 2,480 mushrooms (not recorded). We keep it as its own value,
  `missing`, rather than dropping those rows.
- `veil-type` is `p` (partial) for every mushroom, so it is dropped: it cannot separate anything.

The guide itself says there is no simple rule for telling edible from poisonous mushrooms. This
data is a teaching set, not a foraging tool.
