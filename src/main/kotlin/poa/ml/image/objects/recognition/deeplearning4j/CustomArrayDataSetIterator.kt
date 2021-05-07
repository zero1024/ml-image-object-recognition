package poa.ml.image.objects.recognition.deeplearning4j

import org.deeplearning4j.datasets.iterator.DoublesDataSetIterator
import org.nd4j.common.primitives.Pair

class CustomArrayDataSetIterator(
    X: Array<DoubleArray>,
    y: IntArray,
    batchSize: Int
) :
    DoublesDataSetIterator(toIterable(X, y), batchSize)

private fun toIterable(x: Array<DoubleArray>, y: IntArray): Iterable<Pair<DoubleArray, DoubleArray>> {
    var idx = 0
    return object : Iterable<Pair<DoubleArray, DoubleArray>> {
        override fun iterator(): Iterator<Pair<DoubleArray, DoubleArray>> {
            return object : Iterator<Pair<DoubleArray, DoubleArray>> {

                override fun hasNext() = idx < x.size

                override fun next() = Pair.of(x[idx], mapLabel(y[idx++]))


            }
        }

    }
}

private fun mapLabel(label: Int) =
    if (label == 1) doubleArrayOf(0.0, 1.0) else doubleArrayOf(1.0, 0.0)
