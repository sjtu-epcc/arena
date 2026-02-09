
import argparse
import pickle

# Args 
parser = argparse.ArgumentParser()
# Configurations of communication configurations
parser.add_argument("--path", default="none", type=str)
parser.add_argument("--key", default="none", type=str)
parser.add_argument("--only_key", default=False, action='store_true', 
                    help="Whether to only display keys of the dict.")
args = parser.parse_args()

with open(args.path, "rb") as f:
    results = pickle.load(f)

if args.key != "none":
    print(results[args.key])
elif args.only_key:
    for _key in results.keys():
        print(_key)
else:
    print(results)
