package poa.ml.image.objects.recognition

import java.awt.image.BufferedImage
import kotlin.math.max
import kotlin.math.min

class ImageSamplesCollector(
    private val pxlStep: Int = 5,
    private val slideSize: Int = 50
) {

    init {
        assert(slideSize >= pxlStep) { "slide size should be greater than or equal to pxl step" }
    }

    fun collect(image: BufferedImage): List<Sample> {
        val list = mutableListOf<Sample>()
        for (y in 0 until image.height step pxlStep) {
            for (x in 0 until image.width step pxlStep) {
                val resultX = min(x, image.width - slideSize)
                val resultY = min(y, image.height - slideSize)
                val sub = image.getSubimage(
                    resultX,
                    resultY,
                    slideSize,
                    slideSize
                )
                list.add(Sample(resultX, resultY, sub))
            }
        }
        return list
    }

}

