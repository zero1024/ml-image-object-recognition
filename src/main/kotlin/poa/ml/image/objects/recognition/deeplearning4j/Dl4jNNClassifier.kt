package poa.ml.image.objects.recognition.deeplearning4j

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.CollectScoresListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import smile.classification.Classifier

class Dl4jNNClassifier(
    X: Array<DoubleArray>,
    y: IntArray,
    nEpochs: Int = 400
) : Classifier<DoubleArray> {

    private var model: MultiLayerNetwork
    val scores = CollectScoresListener(10)

    init {
        val batchSize = 100
        val seed = 123
        val learningRate = 0.005
        //Number of epochs (full passes of the data)
        val numInputs = 10800
        val numOutputs = 2
        val numHiddenNodes = 20

        val trainIter = CustomArrayDataSetIterator(X, y, batchSize)

        val conf: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .weightInit(WeightInit.XAVIER)
            .updater(Nesterovs(learningRate, 0.9))
            .list()
            .layer(
                DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                    .activation(Activation.RELU)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX)
                    .nIn(numHiddenNodes).nOut(numOutputs).build()
            )
            .build()
        model = MultiLayerNetwork(conf)
        model.init()
        model.setListeners(scores)
        model.fit(trainIter, nEpochs)
    }

    override fun predict(x: DoubleArray): Int {
        return model.predict(Nd4j.create(arrayOf(x)))[0]
    }


}