# HIM overlay

This additive overlay wraps the repository's existing policy boundary. Keep native
source, environments, trainers, datasets, and checkpoints outside `.him/`.

Run `him validate`, `him test --local`, and `him pack --check`. Review the printed
private upload inventory before any `him push --private --yes` command.
