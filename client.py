class LoopyBeliefPropagation:
    """Sum-Product Loopy Belief Propagation on pairwise factor graphs."""
    def run_lbp(self, num_nodes: int, node_potentials: list[list[float]],
                edges: list[tuple[int, int]], edge_potentials: dict,
                max_iters: int = 15, damping: float = 0.5) -> dict:
        # Messages from u to v: m_{u->v}[state]
        messages = {}
        for u, v in edges:
            messages[(u, v)] = [1.0, 1.0]
            messages[(v, u)] = [1.0, 1.0]

        for _ in range(max_iters):
            new_messages = {}
            for u, v in edges:
                for direction in [(u, v), (v, u)]:
                    src, dst = direction
                    # Compute outgoing message
                    factor_matrix = edge_potentials.get((min(src, dst), max(src, dst)), [[1, 0.5], [0.5, 1]])
                    out_msg = [0.0, 0.0]
                    for x_dst in [0, 1]:
                        s = 0.0
                        for x_src in [0, 1]:
                            pot = node_potentials[src][x_src]
                            pair_pot = factor_matrix[x_src][x_dst] if src < dst else factor_matrix[x_dst][x_src]
                            prod_incoming = 1.0
                            for neighbor in [n1 for (n1, n2) in messages if n2 == src and n1 != dst]:
                                prod_incoming *= messages[(neighbor, src)][x_src]
                            s += pot * pair_pot * prod_incoming
                        out_msg[x_dst] = s

                    # Normalize & Damp
                    tot = sum(out_msg) or 1.0
                    norm_msg = [x / tot for x in out_msg]
                    old_msg = messages[direction]
                    damped = [damping * old_msg[i] + (1 - damping) * norm_msg[i] for i in range(2)]
                    new_messages[direction] = damped
            messages.update(new_messages)

        # Beliefs
        beliefs = []
        for i in range(num_nodes):
            b = list(node_potentials[i])
            for (src, dst), msg in messages.items():
                if dst == i:
                    b[0] *= msg[0]
                    b[1] *= msg[1]
            tot = sum(b) or 1.0
            beliefs.append([round(b[0] / tot, 4), round(b[1] / tot, 4)])

        return {
            "num_nodes": num_nodes,
            "converged_beliefs": beliefs
        }
