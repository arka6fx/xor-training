import numpy as np

#XOR input 
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1] 
])

#XOR output
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

#XOR dataset
print("Input -> Output")
for i in range(len(X)):
    print(f"{X[i]} -> {Y[i][0]}")


#network architecture
INPUT_SIZE = 2
HIDDEN_SIZE  = 4
OUTPUT_SIZE = 1

np.random.seed(42)

#weights from input to hidden layer and a bias term for each hidden neuron
weights_input_hidden = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
bias_hidden = np.zeros((1, HIDDEN_SIZE))

#weights from hidden to output layer and a bias term for each output neuron
weights_hidden_output = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5
bias_output = np.zeros((1, OUTPUT_SIZE))

print("Network initialized with random weights and bias values")
print(f"Input -> Hidden weight shape : {weights_input_hidden.shape}")
print(f"Hidden -> Output weight shape : {weights_hidden_output.shape}")
print(f"Total parameters : {weights_input_hidden.size + bias_hidden.size + weights_hidden_output.size + bias_output.size}")

#activation function 
def sigmoid(x):
    """squashes values in the range (0, 1)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

"""
forward pass (flowing of data through the network):
    Step 1. Input -> Hidden : Multiply inputs by weights, add bias, apply activation function
    Step 2. Hidden -> Output : Multiply hidden by weights, add bias, apply activation function

"""
def forward(X):
    #step 1
    z_hidden = np.dot(X, weights_input_hidden) + bias_hidden
    a_hidden = sigmoid(z_hidden)
    #step 2
    z_output = np.dot(a_hidden, weights_hidden_output) + bias_output
    a_output = sigmoid(z_output)

    return z_hidden, a_hidden, z_output, a_output


""" 
loss function (mean squared error)
Loss measures how wrong our predictions are. Lower = better.
"""

def calculate_loss(prediction, true):
    """mean squared loss"""
    return np.mean((prediction - true) ** 2)


"""
Backpropagation : 
This is where the magic happens. Backpropagation answers: "which weights caused the error and how much?"
The Chain of Blame:
 1. calculate error at the output
 2. Figure out how much each output weight contributed
 3. propagate error back to hidden layer
 4. Figure out how much each hidden weight contributed
 5. Adjust all weights proportionally
 The math uses the chain rule of calculus, but the intuition is simple: blame flows backward
"""

def backward(X, Y, z_hidden, a_hidden, z_output, a_output, learning_rate):
    """ compute gradients and update weights """
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

    m = X.shape[0]

    # 1. calculate error at the output layer (prediction - true value)
    output_error = a_output - Y

    #2. Figure out how much each output weight contributed

        # Gradient of loss w.r.t. z_output (before activation)
        # This combines the error with the sigmoid derivative
    output_delta = output_error * sigmoid_derivative(z_output)

        #Gradient of loss w.r.t  weights_hidden_output
    grad_weights_hidden_output = np.dot(a_hidden.T, output_delta) / m
    grad_bias_output = np.mean(output_delta, axis=0, keepdims=True)

    #3. propagate error back to hidden layer
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
 
    #4. Figure out how much each hidden weight contributed

        #Gradient of loss w.r.t z_hidden
    hidden_delta = hidden_error * sigmoid_derivative(z_hidden)    
        
        #Gradient of loss w.r.t weights_input_hidden
    grad_weights_input_hidden = np.dot(X.T, hidden_delta) / m
    grad_bias_hidden = np.mean(hidden_delta, axis=0, keepdims=True)

    #5. Adjust all weights proportionally

        #move weights in the opposite direction of the gradient 
        #gradient points uphill, we want to go downhill
    weights_input_hidden -= learning_rate * grad_weights_input_hidden
    bias_hidden -= learning_rate * grad_bias_hidden
    weights_hidden_output -= learning_rate * grad_weights_hidden_output
    bias_output -= learning_rate * grad_bias_output

"""
Training Loop:

for each iteration:
    1. Forward pass -> get prediction
    2. Calculate loss -> how wrong we are
    3. Backward pass -> compute gradients, update weights
"""

#Hyperparameters
learning_rate = 2.0
iterations = 10000

print("Training started...")
print("-" * 50)

for i in range(iterations + 1):
    #Forward pass
    z_h, a_h, z_o, predictions = forward(X)

    #calculate loss
    loss = calculate_loss(predictions, Y)

    #Backward pass
    backward(X, Y, z_h, a_h, z_o, predictions, learning_rate)

    if i % 1000 == 0:
        print(f" Iteration: {i} | Loss: {loss:.6f}")

#final prediction 
_, _, _, final_predictions = forward(X)

print("Final Results After Training:")
print("-" * 50)
print(f"{'Input':<12} {'Target':<10} {'Prediction':<10} {'Rounded':<12} {'status':<10}")
for i in range(len(X)):
    pred = final_predictions[i][0]
    rounded = round(pred)
    status = "✅" if rounded == Y[i][0] else "❌"
    print(f"{str(X[i]):<12} {Y[i][0]:<10} {pred:<12.4f} {rounded:<10} {status}")
