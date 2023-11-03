# De-sounding Echo Chambers

An exploration of social polarization dynamics and echo chambers within online social networks.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [File Descriptions](#file-descriptions)
- [Setup and Installation](#setup-and-installation)
- [Running the Simulation](#running-the-simulation)
- [Evaluating Results](#evaluating-results)
- [Contributing](#contributing)
- [Creators](#creators)
- [License](#license)

## Quick Start

To get started with the simulation, you need to set up the project environment using Poetry, ensure the availability of CUDA for performance enhancement, generate or supply the required dataset, and run the simulation using provided instructions.

## Prerequisites

- Python 3.8 or later
- Poetry for Python package management and dependency resolution
- A CUDA-compatible GPU and corresponding toolkit for improved performance (optional but recommended)

## File Descriptions

The simulation relies on four structured data files to model the social network and its dynamics. Below is the expected format and description of each file:

1. `community_info.csv` - This file contains details about the identified communities within the social network, with columns for community labels, unique identifiers, ideological alignment (where `1` signifies an ideological community and `0` indicates no ideological behavior), and member count. 

    **Example Entry:**
    ```
    label, id, ideological, count
    red, 44123, 1, 2550
    blue, 44124, 0, 2550
    ```

2. `edge_info.csv` - Provides metadata about the different types of edges or relationships that can exist within the social network graph, including post, repost, membership, and follow relationships.

    **Example Entry:**
    ```
    edge_type, count, query_type, head_1_type, head_2_type, tail_type, dependency
    post, 39023, 1-chain, tweet, , user, ()
    repost, 130569, 1-chain, user, , tweet, ()
    member, 5100, 1-chain, user, , community, ()
    follow, 31212, 2-chain, user, , user, ('repost', 'post')
    post_member, 39023, 2-chain, tweet, , community, ('post', 'member')
    ```

3. `edges.csv` - Captures the actual edges between nodes in the network, specifying the type of relationship and the entities involved.

    **Example Entry:**
    ```
    ,head, tail, edge_type, query_type, head_type, tail_type
    0,7677,412,post,1-chain,tweet,user
    1,12761,67,post,1-chain,tweet,user
    2,23189,2991,post,1-chain,tweet,user
    3,39669,4928,post,1-chain,tweet,user
    4,42738,3697,post,1-chain,tweet,user
    ...
    ```

4. `node_info.csv` - Outlines the types of nodes present in the network and their quantities.

    **Example Entry:**
    ```
    node_type, count
    user, 5100
    tweet, 39023
    community, 2
    ```

These files should be placed in the appropriate directory as specified by your simulation's configuration. If you wish to use your own dataset, ensure it adheres to the structure demonstrated above to maintain compatibility with the simulation framework.


## Setup and Installation

To set up the project, follow these steps:

1. Install Poetry:
    ```sh
    pip install poetry
    ```

2. Clone the repository and navigate to the project directory:
    ```sh
    git clone https://github.com/ti-ra-do/de-sounding-echo-chambers.git
    cd de-sounding-echo-chambers
    ```

3. Install dependencies using Poetry:
    ```sh
    poetry install
    ```

4. If you have a CUDA environment, ensure it is properly set up to accelerate computation.

## Running the Simulation

After setting up the project, generate synthetic datasets or use existing datasets that conform to the described specifications. Run the simulation with:

```sh
poetry run python src/main.py --dataset synth_polarization --condition <CONDITION> --confirmation_bias <BIAS> [--cuda]
```

Replace `<CONDITION>` with one of the following conditions: `epistemic`, `ideological`, or `conditional_ideological`. Set `<BIAS>` to a value between `0.0` (no acceptance) and `1.0` (full acceptance). Include `--cuda` if you are utilizing a CUDA environment for improved performance.

## Evaluating Results

To track and evaluate the state of the simulation:

1. Launch JupyterLab via Poetry:
   ```sh
   poetry run jupyter lab
2. Navigate to the notebooks/ directory and open either evaluate.ipynb or evaluate_multiple.ipynb.

## Contributing

As this is still a work in progress, contributions to this project are greatly appreciated!

## Creators

Tim Donkers

    https://github.com/ti-ra-do

## License

This project is open source and available under the MIT License.