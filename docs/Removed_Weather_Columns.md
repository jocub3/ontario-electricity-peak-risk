# Removed Weather Columns

**Analysis period:** 2021-01-01 through 2025-12-31.

The source weather files remain unchanged. During Master Dataset construction,
only columns that were completely empty across the combined, date-filtered
weather dataset were removed. Partially populated fields were retained.

## Removed columns

| column              | reason                                                     |   missing_pct |
|:--------------------|:-----------------------------------------------------------|--------------:|
| Flag                | 100% missing across the combined 2021-2025 weather dataset |           100 |
| Temp Flag           | 100% missing across the combined 2021-2025 weather dataset |           100 |
| Precip. Amount Flag | 100% missing across the combined 2021-2025 weather dataset |           100 |
| Wind Dir Flag       | 100% missing across the combined 2021-2025 weather dataset |           100 |
| Wind Spd Flag       | 100% missing across the combined 2021-2025 weather dataset |           100 |
| Stn Press Flag      | 100% missing across the combined 2021-2025 weather dataset |           100 |
| Hmdx Flag           | 100% missing across the combined 2021-2025 weather dataset |           100 |
| Wind Chill Flag     | 100% missing across the combined 2021-2025 weather dataset |           100 |
