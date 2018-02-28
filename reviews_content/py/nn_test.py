
## Code taken from the first course in Coursera's Deep Learning 
## specialization. None of the code in this file belongs to me.

np.random.seed(1)
parameters = initialize_parameters([5,4,3])

A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)

A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = sigmoid)
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = relu)
print("With ReLU: A = " + str(A))

activations = [relu, relu, relu, sigmoid]

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters, activations)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))

Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))

dZ, linear_cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


dAL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = sigmoid)
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = relu)
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches, activations)
print_grads(grads)

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))