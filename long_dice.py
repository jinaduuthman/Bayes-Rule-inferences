import numpy as np
import sys
import itertools

# Check the command-line
if len(sys.argv) != 5:
    print(f"Usage: {sys.argv[0]} <dice> <sides> <input.txt> <out.csv>")
    exit(0)

# Get the command-line arguments
num_dice = int(sys.argv[1])
num_sides = int(sys.argv[2])
infilename = sys.argv[3]
outfilename = sys.argv[4]

# Bounds on the sum of the dice: sum_lb <= possible sum < sum_ub
sum_ub = num_dice * num_sides + 1
sum_lb = num_dice

# Pretty print the lookup table
def show_odds(odds):
    global sum_lb, sum_ub
    print('\n*** Lookup Table ***')
    for i in range(sum_lb, sum_ub):
        print(f"\tGiven d={i}:")
        print(f"\t\t", end="")
        for j in ['H','E','L']:
            print(f"log(P({j}))={odds[j][i]:.5f} ", end="")
        print()

# Pretty print entries of 'array'
def show_array(label, array):
    global sum_lb, sum_ub
    print(f"\t{label}: ", end="")
    for i in range(sum_lb, sum_ub):
        print(f"{i}:{array[i]:.4f} ", end="")
    print()
    
# Recursive function to fill an array with integers
# how_many_ways[8] holds number of ways n dice (each with m sides) make the number 8
def fill_how_many_ways(ndice, nsides, how_many_ways):
    for rolls in itertools.product(range(1,nsides+1), repeat=ndice):
        how_many_ways[sum(rolls)] += 1

print("*** Dice ***")
how_many_ways = np.zeros(sum_ub, dtype=np.int32)
fill_how_many_ways(num_dice, num_sides, how_many_ways)
show_array('How many ways', how_many_ways)

# What is the probability of getting a sum of n when you roll the dice?
priors = how_many_ways / (num_sides ** num_dice)
show_array('Priors', priors)

# Get the log of the priors
log_priors = np.log(priors)

# Create a lookup table
# Example: log_odds['H'][8] holds the log of the probability of an 'H' given the original sum is 8
log_odds = {'H': np.zeros(sum_ub), 'E': np.zeros(sum_ub), 'L': np.zeros(sum_ub)}

# Step through every possible value for the original sum
for original_sum in range(0, sum_ub):
    h_sum = 0.0
    l_sum = 0.0
    for j in range(sum_lb, sum_ub):
        if original_sum == j:
            log_odds['E'][original_sum] = log_priors[j]
        elif original_sum < j:
            h_sum += priors[j]
        else:
            l_sum += priors[j]
    if h_sum > 0:
        log_odds['H'][original_sum] = np.log(h_sum)
    else:
        log_odds['H'][original_sum] = -np.inf
    if l_sum > 0:
        log_odds['L'][original_sum] = np.log(l_sum)
    else:
        log_odds['L'][original_sum] = -np.inf

# Show the lookup table
show_odds(log_odds)

# Open the output file
outfile = open(outfilename,'w')

# Write the header
outfile.write("input,")
for i in range(sum_lb, sum_ub):
    outfile.write(f"P(d={i}),")
outfile.write("guess\n")

# Open the input file
with open(infilename, 'r') as f:

    # Step through each line in the input file
    for row in f.readlines():

        # Remove newline
        row = row.strip()

        # Shorten long inputs to just 7 characters
        if len(row) <= 7:
            out_row = row
        else:
            out_row = row[:4] + "..."

        # Is this an empty line?
        if len(row) == 0:
            continue

        # print(f"\n*** Input:\"{row}\" ***")

        # Compute the likelihoods
        # Example: log_likelihoods[9] holds the log of the probability of 'row' given the original sum is 9
        log_likelihoods = np.zeros(sum_ub, dtype=np.float64)
        for r in row:
            log_likelihoods += log_odds[r]
        # show_array("Log Likelihoods", log_likelihoods)

        # Bayes rule: the posterior is proportional to the likelihood times the prior
        unnormalized_log_posteriors = log_likelihoods + log_priors
        # show_array("Unnormalized log posteriors", unnormalized_log_posteriors)

        # Which is the most likely explanation?
        best = np.argmax(unnormalized_log_posteriors)
        # print(f"\tMAP: {best}")

        # But they don't sum to 1.0, so we need scale them so they do
        # (Includes subtracting the max which can help prevent underflow)
        unnormalized_posteriors = np.exp(unnormalized_log_posteriors - np.max(unnormalized_log_posteriors))
        # show_array("Unnormalized posteriors", unnormalized_posteriors)
        
        # Scale so they sum to 1.0
        normalized_posteriors = unnormalized_posteriors / np.sum(unnormalized_posteriors)
        # show_array("Normalized posteriors", normalized_posteriors)

        # Write them out
        outfile.write(f"{out_row},")
        for i in range(sum_lb, sum_ub):
            outfile.write(f"{normalized_posteriors[i]:.5f},")

        # Which is the most likely explanation?
        best = np.argmax(normalized_posteriors)
        outfile.write(f"{best}\n")

# Close the output file
outfile.close()
