# Community Software Analysis Proposal

## Software: Firedrake

*Write a paragraph describing what the software does and who its
primary audience is.*

Firedrake is an open-source finite-element software package for solving PDEs. It is very similar to FEniCS which takes advantage of UFL to describe the finite element discretization of PDEs. The main difference is that Firedrake uses PyOP2 rather than DOLFIN/DOLFINx which is slightly more efficient for a Poisson and N-S benchmarks (https://fenicsproject.org/pub/presentations/fenics14-paris/FEniCS14FlorianRathgeber.pdf). It is a more activity community than FEniCS. Its primary audience is mathematicians and engineers.

### Stats

| Description | Your answer |
|---------|-----------|
| Repository URL | https://github.com/firedrakeproject/firedrake |
| Main/documentation website | https://www.firedrakeproject.org/index.html |
| Year project was started | 2013 |
| Number of contributors in the past year | 40 |
| Number of contributors in the lifetime of the project | 82 |
| Number of distinct affiliations | > 10 |
| Where do development discussions take place? | Github issues |
| Typical number of emails/comments per week? | 1 |
| Typical number of commits per week? | 3-4 |
| Typical commit size | 20-500 insertions |
| How does the project accept contributions? | pull requests? |
| Does the project have an automated test suite? | yes |
| Does the project use continuous integration? | yes |
| Are any legal/licensing steps required to contribute? | N/A |

Contributions not clear, no documentation.

### Install and run

Check the following boxes when complete or add a note below if you
encountered a problem.

- [x] I have installed the software
- [x] I have run at least one example
- [x] I have run the test suite
- [x] The test suite passes

### Notes/concerns/risks

Concern for proper SOP for contributing.

#### Note on copyright
Students retain copyright on any work done in completion of a CU
course, so you are authorized to sign a [contributor license
agreement (CLA)](https://en.wikipedia.org/wiki/Contributor_License_Agreement),
affirm a [developer's certificate of
origin (DCO)](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin),
etc.  If you have concerns about this, please note them and/or reach
out to Jed directly.
