# Development

This document will (loosely) keep track of development goals for this project.

---

## Current Progress

The linear algebra library previously in rusty-machine is now a new crate - [Rulinalg](https://github.com/AtheMathmo/rulinalg).

For full information on what is currently available look at the [crate documentation](https://athemathmo.github.io/rusty-machine/rusty-machine/doc/rusty_machine/index.html).

---

## Goals

The table below details some planned features and the release version we are aiming for.
We are actively developing and so expect to move through these at a good pace!

<table>
    <tr>
        <th>Version</th><th>Feature</th><th>Dependencies</th>
    </tr>
    <tr>
        <td>0.5.X</td><td>Nearest Neighbours</td><td><ul><li>None</li></ul></td>
    </tr>
    <tr>
        <td>0.6.0</td><td>Model API Improvements</td><td><ul><li>None</li></ul></td>
    </tr>
    <tr>
        <td>0.6.0</td><td>Neural Net Improvements</td><td><ul><li>None</li></ul></td>
    </tr>
</table>

Whilst working on the above there will of course be ongoing development on the existing and new machine learning algorithms.

### Unplanned:

- Convolutional and Recurrent neural nets.
- SVM coordinate descent as per [this paper](http://www.loshchilov.com/publications/GECCO2011_AdaptiveCoordinateDescent.pdf).

### Why aren't you working on X?

If you think there is an obvious feature missing from the library please open an issue about it.
If you want to work on said feature then even better!
