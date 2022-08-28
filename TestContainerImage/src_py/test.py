#import sys
import cv2

print(cv2.getBuildInformation())
print(cv2.cuda.getCudaEnabledDeviceCount())

src = cv2.imread("./resource/lena.jpg",cv2.IMREAD_GRAYSCALE)

#CPU
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
dst1 = clahe.apply(src)
cv2.imshow('CPU', dst1)

#GPU
src_gpu = cv2.cuda_GpuMat()
src_gpu.upload(src)
clahe = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
dst_gpu = clahe.apply(src_gpu, cv2.cuda_Stream.Null())
dst2 = dst_gpu.download()

#Show
cv2.imshow("GPU", dst2)
cv2.waitKey(5000)
cv2.destroyAllWindows()
