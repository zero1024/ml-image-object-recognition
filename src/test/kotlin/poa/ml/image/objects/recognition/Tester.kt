package poa.ml.image.objects.recognition

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

class Tester {

    private val imageSamplesCollector = ImageSamplesCollector(
        pxlStep = 100,
        slideSize = 100
    )                
    private val imageSampleLabeler = ImageSampleLabeler()

    @Test
    internal fun testAllTheThings() {

        runBlocking {
            val testImage = testImage()
            val samples = imageSamplesCollector.collect(testImage)
            val labeledSamples = imageSampleLabeler.label(samples)
            println(labeledSamples)

        }


    }
}


private fun testImage(): BufferedImage {
    return ImageIO.read(File("/Users/oleg1024/Downloads/iriska.jpg"))
}
