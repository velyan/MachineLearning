/*:
# Ordinary Least Squares (OLS)
 
### A linear regression with a least-squares penalty.
 */
import Foundation
//:## Defining the training set
//: Variables **x** and **y** contain the features and labels of the training set respectively. The intercept is 1.
let x: [[Double]] = [[1, 50], [1, 76], [1, 26], [1, 102]]
let y: [Double] = [30, 48, 12, 90]

//:## Prediction function
//: The *predict* function takes as parameter the feature vector **x** and the model **theta** and returns the predicted label.
func predict(x: [Double], theta: [Double]) -> Double {
    let yHat = x[0] * theta[0] + x[1] * theta[1]
    return yHat
}

//:## Defining the cost function
//:### Cost function on a single sample
//: A function *cost* that takes as parameter the predicted label **y** and the actual label **yHat** of a single sample and returns the value of the cost function for this pair.
func cost(yHat: Double, y: Double) -> Double {
    return pow(yHat - y, 2) / 2
}

//:### Cost function on the whole training set
//: We can easily compute the cost function for the whole training set by summing the cost function values for all the samples in the training set.
func costTotal(x: [[Double]], y: [Double], theta: [Double]) -> Double {
    var total = 0.0
    
    for (idx, x) in x.enumerated() {
        let yHat = predict(x: x, theta: theta)
        let y = y[idx]
        let loss = cost(yHat: yHat, y: y)
        total += loss
    }
    return total
}

let cost = costTotal(x: x, y: y, theta: [0,0])

//:## Defining the gradient of the cost function
//:### Gradient on a single sample
//: A function *gradient* that implements the gradient of the cost function for a given sample _(x, y)_.
func gradient(x: [Double], y: Double, theta: [Double]) -> [Double] {
    var gradient: [Double] = []
    let yHat = predict(x: x, theta: theta)
    x.forEach({ x in
        let value = (yHat - y) * x
        gradient.append(value)
    })
    return gradient
}

//:### Gradient on the whole training set
//: We can easily compute *gradientTotal*, the gradient of the cost function on the whole training set by summing the gradients for all the samples in the training set.
func gradientTotal(x: [[Double]], y: [Double], theta: [Double]) -> [Double] {
    var grad = [0.0, 0.0]
    for (idx, x) in x.enumerated() {
        let y = y[idx]
        let value = gradient(x: x, y: y, theta: theta)
        grad = zip(grad, value).map(+)
    }

    return grad
}

let gradTotal = gradientTotal(x: x, y: y, theta: [0,0])

//:## Applying a gradient descent
//:### Gradient descent step implementation
//: The *gradientDescentStep* implements a single gradient step of the gradient descent algorithm.
func gradientDescentStep(x: [[Double]], y: [Double], theta: [Double], alpha: Double) -> [Double] {
    let grad = gradientTotal(x: x, y: y, theta: theta)
    return zip(theta, grad).map({ $0 - alpha * $1 })
}

//:### Iterating gradient descent steps
//: The *gradientDescent* implentation starts from a given **theta** (example _theta = [0, 0]_) and applies 100 gradient descent iterations.
func gradientDescent(x: [[Double]], y: [Double], theta: [Double], alpha: Double) -> [Double] {
    var thetaTrained = theta
    for _ in 0 ..< 100 {
        thetaTrained = gradientDescentStep(x: x, y: y, theta: thetaTrained, alpha: alpha)
        let cost = costTotal(x: x, y: y, theta: thetaTrained)
        print(cost)
    }
    return thetaTrained
}

let alpha = 0.0001
let theta = [0.0, 0.0]
gradientDescent(x: x, y: y, theta: theta, alpha: alpha)

//:## Stochastic gradient descent
//:### Stochastic gradient descent step
//: Function *stochasticGradientDescentStep* computes a stochastic gradient step (on a single _(x, y)_ sample)
func stochasticGradientDescentStep(x: [Double], y: Double, theta: [Double], alpha: Double) -> [Double] {
    let grad = gradient(x: x, y: y, theta: theta)
    return zip(theta, grad).map({ $0 - alpha * $1 })
}

let stochasticStep = stochasticGradientDescentStep(x: x.first!, y: y.first!, theta: theta, alpha: alpha)

//:### Stochastic gradient descent implementation
//: Function *stochasticGradientDescent* iterates 100 stochastic gradient descent steps and returns the trained model **theta**.
func stochasticGradientDescent(x: [[Double]], y: [Double], theta: [Double], alpha: Double) -> [Double] {
    var thetaTrained = theta
    let allX = x
    let allY = y

    for _ in 0..<100 {
        for (idx, x) in x.enumerated() {
            let y = y[idx]
            thetaTrained = stochasticGradientDescentStep(x: x, y: y, theta: thetaTrained, alpha: alpha)
            let cost = costTotal(x: allX, y: allY, theta: thetaTrained)
            print(cost)
        }
    }
    return thetaTrained
}

let trainedTheta = stochasticGradientDescent(x: x, y: y, theta: theta, alpha: alpha)//gradientDescentStep(x: x, y: y, theta: theta, alpha: alpha)/

//:## Regression
func regression(theta: [Double]) -> [Double] {
    var values: [Double] = []
    for i in 0..<140 {
        var value = 0.0
        for (idx, x) in x.enumerated() {
            if x.last! == Double(i) {
                value = y[idx]
            }
        }
        //Expand this line to observe the obtained regression.
        value = value > 0 ? value : predictedValue(x: Double(i), theta: theta)
        values.append(value)
    }
    return values
}

func predictedValue(x: Double, theta: [Double]) -> Double {
    return theta.first! + x * theta.last!
}

regression(theta: trainedTheta)
