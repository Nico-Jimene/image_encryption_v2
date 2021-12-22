#import os and sys to make sure file writes in the correct location
import os
import sys
#set file write directory to location of this file
os.chdir(os.path.dirname(sys.argv[0]))

#importing other functions for use throughout the project
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#gets a string with a tiff file extension
def GetTiffExtension(inputText):
    fileName = input(inputText)
    while os.path.splitext(fileName)[1] != ".tiff":
        fileName = input(f"File extension must be .tiff\n{inputText}")
    return fileName

#given a file name, imports the relevant file and makes sure that the file is in the right format, etc
def ImportTiff(fileName):
    try:
        image = plt.imread(fileName)[:,:,:3].astype("uint8")
    except:
        print("Something went wrong. Please make sure file exists")
        sys.exit()
    return image

#given info about an image and a keystring, returns a 2-d array with generated values used to 
#encrypt the image
def KeyGenerator(height, width, keyString):
    numLetters = len(keyString.replace(" ",""))
    keyArray = np.empty(shape=(height,width), dtype="uint8")
    for row in range(height):
        for col in range(width):
            keyArray[row,col] = (row * col) % numLetters * (2**8//numLetters)

    return keyArray

#a better generator that uses a random number generator where the random seed is based on the keystring
#this results in an image with more entropy that is harder to decrypt
def NewKeyGenerator(height,width,keyString):
    seedNum = len(keyString.replace(" ",""))
    np.random.seed(seedNum)
    keyArray = np.random.randint(255, size = height * width).reshape(height,width)

    return keyArray

#given a 3-d image and 2-d key, uses the binary XOR function to either encrypt or decrypt the image
#using the key
def ImageBinaryXOR(image, key):
    for color in range(3):
        #who knew numpy could do magic like this
        image[:,:,color] = image[:,:,color] ^ key
    return image    

#a convenient function for plotting an image that can also have titles, subtitles, etc
def PlotImage(image, title, subtitle = "", cmap="viridis"):
    plt.imshow(image, cmap=cmap)
    plt.suptitle(title)
    plt.title(subtitle, fontsize = 10)
    plt.axis("off")
    plt.show()

#creates a histogram of the color intensities for an image's red, green, and blue values
#and also creates a title
def PlotHistogram(image, title):
    colors = ["red", "green", "blue"]
    for i, color in enumerate(colors):
        plt.hist(image[:,:,i].reshape(image.shape[0]*image.shape[1]), bins=np.arange(2**8+1), color=color)
    plt.title(title)
    plt.show()

#uses ITU-R BT.601 to convert a 3-d image into a 2-d grayscale version
def GrayScale(image):
    colorMods = [0.299, 0.587, 0.114]
    grayScale = colorMods[0] * image[:,:,0] + colorMods[1] * image[:,:,1] + colorMods[2] * image[:,:,2]
    return grayScale

#uses scipy ndimage libary to apply a gaussian filter to a 2-d image and returns the image
def Smoothing(image):
    image = ndimage.gaussian_filter(image, sigma=2, order=0)
    return image

#detects the edges of an image using the sobel operator to find maximums of the gradients
#in both the x and y directions separately, then computes the magnitude of the vector
def EdgeDetection(image):
    dx = ndimage.sobel(image, 1)
    dy = ndimage.sobel(image, 0)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag

#given a 2-d numpy array, returns the location of the highest/ "brightest" value
def FindBrightestGradientLocation(image):
    indexOfMax = np.unravel_index(image.argmax(), image.shape)
    return indexOfMax

#given an image, center location, and a radius, creates a cropped image centered about
#the center with the correct radius
def CropImage(image, center, radius):
    yIndex = center[0]
    xIndex = center[1]
    
    t = yIndex - radius
    b = yIndex + radius
    l = xIndex - radius
    r = xIndex + radius
    cropped = image[t:b, l:r]
    return cropped

#runs through an example using the Pale Blue Dot image
def PaleBlueDot():
    #import the image and get basic image info
    imageName = GetTiffExtension("Please enter file name for Pale Blue Dot image: ")
    image = ImportTiff(imageName)

    outputName = GetTiffExtension("Please enter output name for cropped view of earth: ")
    keyString = input("Please enter your keystring: ")
    #image = ImportTiff(r"img\Pale_Blue_Dot_Encrypted.tiff")
    height = image.shape[0]
    width = image.shape[1]

    #plot given image with dimensions
    PlotImage(image, "Encrypted Image Given:", f"width: {width}px, height: {height}px")

    #using given keyString, generate an image key and decrypt the image
    key = KeyGenerator(height, width, keyString)
    decryptedImage = ImageBinaryXOR(image, key)

    #plot the decrypted image and its pixel intensity histogram
    PlotImage(decryptedImage, "Decrypted image using following key:", f"{keyString}")
    PlotHistogram(decryptedImage, "Decrypted Image Pixel Intensities")

    #create a grayscale version of the decrypted image and plot it
    grayScaleImage = GrayScale(decryptedImage).astype("float64")
    PlotImage(grayScaleImage, "Grayscale Image", cmap="gray")

    #smooth out the grayscale image and plot it
    smoothedImage = Smoothing(grayScaleImage)
    PlotImage(smoothedImage, "Smoothed Image", cmap="gray")

    #find the edges of the smoothed image and plot it
    edgeDetectionImage = EdgeDetection(smoothedImage)
    PlotImage(edgeDetectionImage, "Image With Edge Detection", cmap="gray")

    #find the location of the earth
    locationOfEarth = FindBrightestGradientLocation(edgeDetectionImage)

    #crop the image of the earth, plot it, then save it
    croppedImage = CropImage(decryptedImage, locationOfEarth, 50)
    PlotImage(croppedImage, "Cropped View of Image Centered About Earth", f"The earth is at x={locationOfEarth[1]}, y={locationOfEarth[0]} relative to the top left of the original image.")
    plt.imsave(outputName, croppedImage)

    #using the same keystring as before, create a new image key using our new key generator
    newkey = NewKeyGenerator(height, width, keyString)
    encryptedImage = ImageBinaryXOR(image,newkey)

    #plot the newly encrypted image and its pixel intensity histogram
    PlotImage(encryptedImage, "Newly encrypted image using New Key Generator and key:", f"{keyString}")
    PlotHistogram(encryptedImage, "Encrypted Image Pixel Intensities")

    #decrypt the newly encrypted image, which shows the user that everything works the same
    newDecryptedImage = ImageBinaryXOR(encryptedImage,newkey)
    PlotImage(newDecryptedImage, "Decrypted Image", "This matches the original decrypted image")

#user inputs info about image in, image out, and keystring. Outputs either encrypted or decrypted image
def EncryptDecrypt():
    #import all info needed
    inputName = GetTiffExtension("Please enter input image name: ")
    inputImage = ImportTiff(inputName)
    outputName = GetTiffExtension("Please enter output image name: ")
    keyString = input("Please enter your keystring: ")

    #creates key using newKeyGenerator
    height = inputImage.shape[0]
    width = inputImage.shape[1]
    key = NewKeyGenerator(height, width, keyString)

    #runs the encryption function and saves the output
    outputImage = ImageBinaryXOR(inputImage, key)
    plt.imsave(outputName, outputImage)

    print(f"Your image has been saved as: {outputName}")

#a menu for deciding which program to run
def main():
    print("Type 1 for example using the Pale Blue Dot image")
    print("Type 2 to encrypt or decrypt your own image")
    num = input("Enter 1 or 2: ")
    if num == "1":
        PaleBlueDot()
    elif num == "2":
        EncryptDecrypt()
    else:
        print("Unrecognized input. Please run program again")

if __name__ == '__main__':
    main()


#Control