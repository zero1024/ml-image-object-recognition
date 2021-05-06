package poa.ml.image.objects.recognition.runner

import poa.ml.image.objects.recognition.printlnEnd
import poa.ml.image.objects.recognition.printlnStart
import poa.ml.image.objects.recognition.readFromFile
import poa.ml.image.objects.recognition.subSet
import smile.base.mlp.Layer
import smile.base.mlp.LayerBuilder
import smile.base.mlp.OutputFunction
import smile.classification.Classifier
import smile.classification.logit
import smile.classification.mlp
import smile.classification.svm
import smile.math.kernel.GaussianKernel
import smile.math.kernel.MercerKernel
import smile.math.matrix.Matrix
import smile.validation.ClassificationValidations
import smile.validation.CrossValidation
import java.util.function.BiFunction

class ModelTrainer {

    fun test(
        trainingSetFile: String,
        classifierCode: String,
        nrows: Int = -1
    ): ClassificationValidations<Classifier<DoubleArray>> {
        val (xArray, yArray) = getTrainingSet(trainingSetFile, classifierCode, nrows)
        return test(xArray, yArray, classifierCode)
    }

    fun test(
        xArray: Matrix,
        yArray: IntArray,
        classifierCode: String
    ): ClassificationValidations<Classifier<DoubleArray>> {
        printlnStart("===Training $classifierCode classifier with ${yArray.size} examples...")
        val classification = CrossValidation.classification(5, xArray.toArray(), yArray, dispatch(classifierCode))
        println("")
        println("")
        println("")
        println(classification.avg)
        println("")
        println("")
        println("")
        printlnEnd("===Done training.")
        return classification
    }

    fun train(
        trainingSetFile: String,
        classifierCode: String,
        nrows: Int = -1
    ): Classifier<DoubleArray> {
        val (xArray, yArray) = getTrainingSet(trainingSetFile, classifierCode, nrows)
        printlnStart("===Training $classifierCode classifier with ${yArray.size} examples...")
        val res = dispatch(classifierCode).apply(xArray.toArray(), yArray)
        printlnEnd("===Done training.")
        return res
    }

    fun getTrainingSet(trainingSetFile: String, classifierCode: String, nrows: Int = -1): Pair<Matrix, IntArray> {
        printlnStart("===Reading the matrix...")
        val X = readFromFile<Matrix>("$trainingSetFile.X")
        printlnEnd("===Done reading. Size = ${X.nrows()}x${X.ncols()}")
        printlnStart("===Reading the labels...")
        val y = readFromFile<IntArray>("$trainingSetFile.y")
        printlnEnd("===Done reading.")
        val ySvm = if (classifierCode.startsWith("svm_"))
            y.map { if (it == 0) -1 else 1 }.toIntArray()
        else y
        return subSet(X, ySvm, nrows)
    }


}

private fun dispatch(name: String): BiFunction<Array<DoubleArray>, IntArray, Classifier<DoubleArray>> {
    return when {
        name.startsWith("mlp_") -> {
            val layers = name.split("_")
                .mapNotNull { it.toIntOrNull() }
                .map { Layer.sigmoid(it) as LayerBuilder }
                .toMutableList()
            layers.add(Layer.mle(2, OutputFunction.SIGMOID))
            BiFunction { X, y -> mlp(X, y, layers.toTypedArray(), weightDecay = 0.03) }
        }
        name.startsWith("logit_") -> {
            val lambda = name.split("_")[1]
            BiFunction { X, y -> logit(X, y, lambda.toDouble()) }
        }
        name.startsWith("svm_") -> {
            val (sigma, C) = name.split("_").let { it[1] to it[2] }
            BiFunction { X, y -> svm(X, y, GaussianKernel(sigma.toDouble()), C.toDouble()) }
        }
        else -> {
            throw IllegalStateException()
        }
    }
}

