import cv2
import numpy as np
from matplotlib import pyplot as plt


# test matrix
# img0 = np.array([[[2, 1, 1, 2],
#                 [2, 0, 2, 2],
#                 [0, 0, 0, 0], 
#                 [1, 1, 1, 1],
#                 [1, 2, 1, 1]]])

# img1 = np.array([
#         [[1, 2, 1], [2, 1, 3], [1, 4, 2], [2, 1, 5], [0, 1, 4]],
#         [[2, 3, 1], [1, 5, 1], [7, 1, 8], [1, 5, 2], [2, 5, 2]],
#         [[2, 3, 1], [1, 5, 1], [7, 1, 8], [1, 5, 2], [2, 5, 2]],
#         [[1, 2, 1], [2, 1, 3], [1, 4, 2], [2, 1, 5], [0, 1, 4]],
#         [[2, 3, 1], [1, 5, 1], [7, 1, 8], [1, 5, 2], [2, 5, 2]],
# ])

# img2 = np.array([
#         [[10, 25, 30], [42, 18, 99], [55, 72, 63], [88, 14, 21], [39, 47, 56]],
#         [[12, 34, 78], [67, 45, 89], [91, 53, 24], [73, 31, 62], [85, 12, 33]],
#         [[22, 44, 66], [99, 77, 55], [88, 11, 19], [42, 63, 74], [15, 26, 37]],
#         [[65, 87, 32], [41, 20, 93], [57, 62, 44], [19, 28, 39], [75, 84, 12]],
#         [[81, 92, 63], [56, 44, 32], [17, 29, 83], [34, 45, 92], [72, 18, 59]],
# ])


def convertToCMY(img):
        # Normalize img
        normalize_img = img / 255.0

        # Convert to CMY
        cmy_normalized = 1 - normalize_img

        # Scale values back to range [0, 255]
        cmy_img = (cmy_normalized * 255).astype(np.uint8)
        
        return cmy_img

def createGLCM(img, angle):
        # Find the maximum intensity in the image
        maxIntensity = np.max(img) + 1 # intensity starts with 0 and we need the number of intensities, so maximum number should be plus 1

        # Initial GLCM
        glcm = np.zeros([maxIntensity, maxIntensity])

        # Initial displacement vector with given angle
        dv = () # displacement vector
        if angle == 0:
                dv = (0, 1)
        elif angle == 45:
                dv = (-1, 1)
        elif angle == 90:
                dv = (-1, 0)

        channel, height, width = img.shape
        allGLCM = []
        
        # Fill GLCM matrix
        for ch in range(channel):
                for r in range(height):
                        for c in range(width):
                                if r+dv[0] > height-1 or c+dv[1] > width-1:
                                        continue
                                firstVal = img[ch, r, c]
                                secondVal = img[ch, r+dv[0], c+dv[1]]                                
                                glcm[firstVal][secondVal] += 1                                
                allGLCM.append(glcm)
        
        return allGLCM

def calculateLBP(img, plotName):
        height, width, channel = img.shape

        neighbors = [[-1, -1], [-1, 0], [-1, 1],[0, 1],[1, 1], [1, 0], [1, -1], [0, -1]]
        lbp = np.zeros([height-2, width-2, channel])
        for ch in range(channel):                
                for r in range(1, height-1):
                        for c in range(1, width-1):                                
                                vector = []
                                for i in neighbors:
                                        if img[r+i[0], c+i[1], ch] >= img[r, c, ch]:
                                                vector.append("1")
                                        else:
                                                vector.append("0")
                                        
                                binary = ''.join(vector)
                                lbp[r-1, c-1, ch] = int(binary, 2)
        
        # calculate channel's histogram
        x = range(int(np.max(lbp)+1))
        
        hists = []
        for ch in range(3):
                histogram = np.zeros(int(np.max(lbp)+1))
                for i in range(lbp.shape[1]):
                        for j in range(lbp.shape[0]):                        
                                histogram[int(lbp[i,j,0])] += 1
                hists.append(histogram)
        
        # Plot image in form of LBP and their histogram in 3 channel
        plt.subplot(2,3,1), plt.imshow(lbp[:,:,0])
        plt.title('Channel C'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,2), plt.imshow(lbp[:,:,1])
        plt.title('Channel M'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,3), plt.imshow(lbp[:,:,2])
        plt.title('Channel Y'), plt.xticks([]), plt.yticks([])

        plt.subplot(2,3,4), plt.bar(x, hists[0])        
        plt.subplot(2,3,5), plt.bar(x, hists[1])        
        plt.subplot(2,3,6), plt.bar(x, hists[2])                

        plt.suptitle(f"Demonstration of channels LBP & their Histogram image {plotName}", fontsize=16)

        plt.show()        

        # Quantization of LBP histogram
        quantizeHists = []
        quantizeIntensity = 8
        x2 = range(quantizeIntensity)
        for hist in hists:                
                s = 0 # step
                histogram = np.zeros(quantizeIntensity)
                for i in range(0, quantizeIntensity):
                        histogram[i] = sum(hist[s:s+32])
                        s += 32
                quantizeHists.append(histogram)        
        
        print(f"************ Feature vector of {plotName} ************")
        for i in range(channel):
                print(f"Channel {i}: {quantizeHists[i]}")
        print("******************************")
        
        # ******* plot quantized histogram *******
        plt.subplot(2,3,1), plt.bar(x, hists[0])
        plt.title('Channel C')
        plt.subplot(2,3,2), plt.bar(x, hists[1])
        plt.title('Channel M')
        plt.subplot(2,3,3), plt.bar(x, hists[2])
        plt.title('Channel Y')

        plt.subplot(2,3,4), plt.bar(x2, quantizeHists[0])             
        plt.subplot(2,3,5), plt.bar(x2, quantizeHists[1])              
        plt.subplot(2,3,6), plt.bar(x2, quantizeHists[2])     

        plt.suptitle(f"Quantized histogram for {plotName}", fontsize=16)       

        plt.show()

        return np.concatenate(quantizeHists)


if __name__ == "__main__":        
        # Read image
        img1 = cv2.imread("assets/tiger1.jpg")
        img2 = cv2.imread("assets/tiger2.jpg")
        img3 = cv2.imread("assets/cat2.jpg")

        # Check for image existence
        assert img1 is not None, "Image not found"
        assert img2 is not None, "Image not found"
        assert img3 is not None, "Image not found"

        # Convert images from BGR to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        
        while True:
                print("\nWelcome to Erfan's project")
                print("1. Convert RGB image to CMY image.")
                print("2. Display GLCM.")
                print("3. Calculate LBP of matrix.")
                print("4. Exit\n")
                opt = int(input("Please select an option:"))

                if opt == 1:
                        # Convert to CMY color space
                        cmy_img1 = convertToCMY(img1)
                        cmy_img2 = convertToCMY(img2)
                        cmy_img3 = convertToCMY(img3)

                        # Plot image in CMY color space
                        plt.subplot(2,3,1), plt.imshow(img1)
                        plt.title('Tiger 1 (RGB)'), plt.xticks([]), plt.yticks([])
                        plt.subplot(2,3,2), plt.imshow(img2)
                        plt.title('Tiger 2 (RGB)'), plt.xticks([]), plt.yticks([])
                        plt.subplot(2,3,3), plt.imshow(img3)
                        plt.title('Cat (RGB)'), plt.xticks([]), plt.yticks([])
                        
                        plt.subplot(2, 3, 4), plt.imshow(cmy_img1)
                        plt.title('Tiger 1 (CMY)'), plt.xticks([]), plt.yticks([])
                        plt.subplot(2, 3, 5), plt.imshow(cmy_img2)
                        plt.title('Tiger 2 (CMY)'), plt.xticks([]), plt.yticks([])
                        plt.subplot(2, 3, 6), plt.imshow(cmy_img3)
                        plt.title('Cat (CMY)'), plt.xticks([]), plt.yticks([])
                        
                        plt.suptitle("All images in RGB & CMY format", fontsize=16)
                        plt.show()

                elif opt == 2:                        
                        # Get the angle for calculation of GLCM
                        flag = True
                        while flag:
                                print("\nThese are two angles that you can choose for GLCM calculation:")
                                print("1. (0, 45)")
                                print("2. (0, 90)")
                                print("3. (45, 90)")
                                print("4. Back")
                                angles = int(input("Please select: "))

                                if angles not in (1,2,3):
                                        print("Please enter the right number")
                                elif angles == 4:
                                        flag = False
                                else:
                                        flag = False
                        
                        glcm1 = np.array([])
                        glcm2 = np.array([])
                        print("************* GLCM of Tiger 1 *************")
                        if angles == 1:
                                print("************* GLCM in angle 0 *************")
                                glcm1 = createGLCM(img1, 0)
                                print(glcm1)

                                print("************* GLCM in angle 45 *************")
                                glcm2 = createGLCM(img1, 45)
                                print(glcm2)
                        elif angles == 2:
                                print("************* GLCM in angle 0 *************")
                                glcm1 = crateGLCm(img1, 0)
                                print(glcm1)

                                print("************* GLCM in angle 90 *************")
                                glcm2 = crateGLCm(img1, 90)
                                print(glcm2)

                        elif angles == 3:
                                print("************* GLCM in angle 45 *************")
                                glcm1 = createGLCM(img1, 45)
                                print(glcm1)

                                print("************* GLCM in angle 90 *************")
                                glcm2 = createGLCM(img1, 90)   
                                print(glcm2)
                        
                elif opt == 3:
                        cmy_img1 = convertToCMY(img1)
                        cmy_img2 = convertToCMY(img2)
                        cmy_img3 = convertToCMY(img3)

                        print("\nGenerating LBP histogram for first image ...")
                        vector1 = calculateLBP(cmy_img1, "Tiger 1")
                        print("\nGenerating LBP histogram for second image ...")
                        vector2 = calculateLBP(cmy_img2, "Tiger 2")
                        print("\nGenerating LBP histogram for third image ...")
                        vector3 = calculateLBP(cmy_img3, "Cat")
                        
                        # Find the similarity of two image with Manhattan distance
                        sum1 = 0
                        sum2 = 0
                        for i in range(8):
                                sum1 += abs(vector1[i] - vector2[i])
                                sum2 += abs(vector1[i] - vector3[i])

                        if sum1 < sum2:
                                print(f"Image 1 is similar to image 2")
                                plt.subplot(1,3,1), plt.imshow(img1)
                                plt.title('Image 1'), plt.xticks([]), plt.yticks([])
                                plt.subplot(1,3,2), plt.imshow(img2)
                                plt.title('Image 2'), plt.xticks([]), plt.yticks([])
                                plt.subplot(1,3,3), plt.imshow(img3)
                                plt.title('Image 3'), plt.xticks([]), plt.yticks([])
                                plt.suptitle("Image 1 is similar to image 2")
                                plt.show()
                        else:
                                print(f"Image 1 is similar to image 3")
                                plt.subplot(1,3,1), plt.imshow(img1)
                                plt.title('Image 1'), plt.xticks([]), plt.yticks([])
                                plt.subplot(1,3,2), plt.imshow(img2)
                                plt.title('Image 2'), plt.xticks([]), plt.yticks([])
                                plt.subplot(1,3,3), plt.imshow(img3)
                                plt.title('Image 3'), plt.xticks([]), plt.yticks([])
                                plt.suptitle("Image 1 is similar to image 3")
                                plt.show()
                                               
                elif opt == 4:
                        print("Goodbye :)")
                        break
        