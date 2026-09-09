from client import LoopyBeliefPropagation

def main():
    print("=== Loopy Belief Propagation on Factor Graphs ===")
    lbp = LoopyBeliefPropagation()
    node_potentials = [[0.8, 0.2], [0.4, 0.6], [0.5, 0.5]]
    edges = [(0, 1), (1, 2), (2, 0)] # Cyclic triangle factor graph
    edge_potentials = {
        (0, 1): [[0.9, 0.1], [0.1, 0.9]],
        (1, 2): [[0.9, 0.1], [0.1, 0.9]],
        (0, 2): [[0.9, 0.1], [0.1, 0.9]]
    }

    res = lbp.run_lbp(3, node_potentials, edges, edge_potentials)
    print("Converged Beliefs:", res)
    assert len(res["converged_beliefs"]) == 3
    assert abs(sum(res["converged_beliefs"][0]) - 1.0) < 1e-3

    print("Loopy Belief Propagation verified successfully!")

if __name__ == "__main__":
    main()
