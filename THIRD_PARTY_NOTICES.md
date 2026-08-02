# Third-party data and software notices

No third-party raw dataset is redistributed in this package. Users must obtain
each dataset from the versioned official record in `DATASETS_AND_LINKS.csv` and
comply with the licence attached to that record.

## Public headed-stud datasets

- Mendeley Data `10.17632/rfrw3z4hs7.2`, `10.17632/nfmhnzbfy9.2` and
  `10.17632/xtg3w85hdr.1` are recorded by their official version pages as
  Creative Commons Attribution 4.0 (`CC BY 4.0`):
  <https://creativecommons.org/licenses/by/4.0/>.
- Zenodo record `10.5281/zenodo.17607687` is recorded as Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 (`CC BY-NC-SA 4.0`):
  <https://creativecommons.org/licenses/by-nc-sa/4.0/>.

The headed-stud application projection includes only aggregate derivatives,
not rows or record identifiers. Aggregates derived solely from the Mendeley
parents remain under `app/results/cc_by_4_0/`. Aggregates containing material
from the Zenodo record remain under
`app/results/cc_by_nc_sa_4_0/` with a dedicated data-licence notice; they require
attribution, non-commercial use and share-alike treatment separate from the
project code licence.

## Software dependencies

Dependency names and versions are recorded in the frozen environment lock.
Their upstream licences remain controlling; a repository-level code licence
does not relicense a dependency or third-party dataset.
