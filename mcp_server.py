import sys
import json
from client import LoopyBeliefPropagation

def handle_request(req):
    method = req.get("method")
    params = req.get("params", {})
    if method == "infer":
        lbp = LoopyBeliefPropagation()
        return lbp.run_lbp(params.get("num_nodes", 2), params.get("node_potentials", []),
                           [tuple(e) for e in params.get("edges", [])], params.get("edge_potentials", {}))
    return {"error": "Unknown method"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_request(req)
        print(json.dumps(res))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
