# SimpleBioNER
use FastText+BLSTM+CRF to do NER on JNLPBA2004, with 0.7011 f-score.

## Data Set

I got the JNLPBA2004 data set by this [link](http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/).

## Time Cost

On a single 2080ti, it takes about an hour to complete the entire process.

## TODO

currently, it is a simple baseline, still lots of work i can do in the future:

- [ ] use character vector
- [ ] replace BIO with BIOES
- [ ] try BERT
- [ ] improve GPU utilization
