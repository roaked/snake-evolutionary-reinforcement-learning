import io, time
import subprocess

C = [2,2]


for i in range(3):
    for m in range (2):
        C[m][i] = i*m


print(C)