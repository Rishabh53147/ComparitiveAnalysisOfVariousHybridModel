import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from hmmlearn import hmm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("RELIANCE_NS.csv")  # Replace with your dataset
stock_prices = data["Adj Close"].values

# Normalize the stock prices between 0 and 1
normalized_prices = (stock_prices - np.min(stock_prices)) / (np.max(stock_prices) - np.min(stock_prices))

# Convert normalized prices to counts (discretize into bins)
num_bins = 10
counts = np.histogram(normalized_prices, bins=num_bins, range=(0, 1))[0]

# Define GA parameters
num_generations = 50
population_size = 50
num_hidden_states = 2  # Number of hidden states for HMM

# Define HMM parameters
num_hidden_states = 2
num_observed_states = num_bins

# Define GA evaluation function
def evaluate(individual):
    # Decode GA individual to HMM parameters
    hmm_params = individual  # Example: [initial_probabilities, transition_matrix, emission_matrix]

    # Train HMM
    hmm_model = hmm.MultinomialHMM(n_components=num_hidden_states)
    hmm_model.startprob_ = hmm_params[0]
    hmm_model.transmat_ = hmm_params[1]
    hmm_model.emissionprob_ = hmm_params[2]
    
    observed_sequences = counts.reshape(-1, 1)  # Reshape data for HMM
    lengths = [len(counts)]
    hmm_model.fit(observed_sequences, lengths=lengths)

    # Evaluate HMM's likelihood on the observed data
    likelihood = hmm_model.score(observed_sequences)
    
    # Calculate predicted closing prices using the trained HMM
    predicted_prices = hmm_model.predict(observed_sequences).reshape(-1, 1)

    # Calculate MAE, MSE, and RMSE
    mae = mean_absolute_error(stock_prices, predicted_prices)
    mse = mean_squared_error(stock_prices, predicted_prices)
    rmse = np.sqrt(mse)

    return (likelihood,)

# Create DEAP toolbox
# Create DEAP toolbox
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3*num_hidden_states)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Create initial population
population = toolbox.population(n=population_size)

# Run the genetic algorithm
algorithm = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3, ngen=num_generations)

# Get the best individual (HMM parameters) after GA runs
best_individual = tools.selBest(population, k=1)[0]

# Decode the best individual to get HMM parameters
best_hmm_params = best_individual 

# Train the final HMM model using the best parameters
final_hmm_model = hmm.MultinomialHMM(n_components=num_hidden_states)
final_hmm_model.startprob_ = best_hmm_params[0]
final_hmm_model.transmat_ = best_hmm_params[1]
final_hmm_model.emissionprob_ = best_hmm_params[2]

# Train the final HMM model using the counts data
observed_sequences = counts.reshape(-1, 1)
lengths = [len(counts)]
final_hmm_model.fit(observed_sequences, lengths=lengths)

# Generate predicted counts using the trained HMM
predicted_counts = final_hmm_model.sample(lengths[0])[0].flatten()

# Decode the predicted counts back to normalized stock prices
decoded_prices = predicted_counts * (np.max(stock_prices) - np.min(stock_prices)) + np.min(stock_prices)

# Calculate the MAE, MSE, and RMSE using the decoded prices
mae = mean_absolute_error(stock_prices, decoded_prices)
mse = mean_squared_error(stock_prices, decoded_prices)
rmse = np.sqrt(mse)

# Print the predicted next state and MAE, MSE, RMSE
print("Predicted next state:", next_state)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

plt.figure(figsize=(12, 6))
plt.plot(stock_prices, label='Actual Prices', color='blue')
plt.plot(decoded_prices, label='Predicted Prices', color='orange')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()
