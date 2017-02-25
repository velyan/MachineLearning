import Foundation

let x: [[Double]] = [[1, 50], [1, 76], [1, 26], [1, 102]]
let y: [Double] = [30, 48, 12, 90]


func predict(x: [Double], theta: [Double]) -> Double {
    let yHat = x[0] * theta[0] + x[1] * theta[1]
    return yHat
}

//Defining the cost function
func cost(yHat: Double, y: Double) -> Double {
    return pow(yHat - y, 2) / 2
}

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

//Defining the gradient of the cost function
func gradient(x: [Double], y: Double, theta: [Double]) -> [Double] {
    var gradient: [Double] = []
    let yHat = predict(x: x, theta: theta)
    x.forEach({ x in
        let value = (yHat - y) * x
        gradient.append(value)
    })
    return gradient
}

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

func gradientDescentStep(x: [[Double]], y: [Double], theta: [Double], alpha: Double) -> [Double] {
    let grad = gradientTotal(x: x, y: y, theta: theta)
    return zip(theta, grad).map({ $0 - alpha * $1 })
}

let alpha = 0.0001
let theta = [0.0, 0.0]

func gradientDescent(x: [[Double]], y: [Double], theta: [Double], alpha: Double) -> [Double] {
    var thetaTrained = theta
    for _ in 0 ..< 100 {
        thetaTrained = gradientDescentStep(x: x, y: y, theta: thetaTrained, alpha: alpha)
        let cost = costTotal(x: x, y: y, theta: thetaTrained)
        print(cost)
    }
    return thetaTrained
}

gradientDescent(x: x, y: y, theta: theta, alpha: alpha)

func stochasticGradientDescentStep(x: [Double], y: Double, theta: [Double], alpha: Double) -> [Double] {
    let grad = gradient(x: x, y: y, theta: theta)
    return zip(theta, grad).map({ $0 - alpha * $1 })
}

let stochasticStep = stochasticGradientDescentStep(x: x.first!, y: y.first!, theta: theta, alpha: alpha)

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

let trainedTheta = gradientDescentStep(x: x, y: y, theta: theta, alpha: alpha)//stochasticGradientDescent(x: x, y: y, theta: theta, alpha: alpha)

func regression(theta: [Double]) -> [Double] {
    var values: [Double] = []
    for i in 0..<140 {
        var value = 0.0
        for (idx, x) in x.enumerated() {
            if x.last! == Double(i) {
                value = y[idx]
            }
        }
        value = value > 0 ? value : predictedValue(x: Double(i), theta: theta)
        values.append(value)
    }
    return values
}

func predictedValue(x: Double, theta: [Double]) -> Double {
    return theta.first! + x * theta.last!
}

regression(theta: trainedTheta)
