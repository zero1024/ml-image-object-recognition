package poa.ml.image.objects.recognition

import java.awt.image.BufferedImage
import java.awt.image.RasterFormatException

class ImageSamplesCollector(
    private val pxlStep: Int = 5,
    private val slideSize: Int = 50
) {

    fun collect(image: BufferedImage): List<Sample> {
        val list = mutableListOf<Sample>()
        for (y in 0 until image.height step pxlStep) {
            for (x in 0 until image.width step pxlStep) {
                try {
                    val sub = image.getSubimage(x, y, slideSize, slideSize)
                    list.add(Sample(x, y, sub))
                } catch (e: RasterFormatException) {
                    //nothing
                }
            }
        }
        return list
    }

}

