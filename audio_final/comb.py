import subprocess

#p1 = subprocess.Popen("python emotions.py", shell=True, 
#p2 = subprocess.Popen("python RecordSound.py", shell=True)

#send w to p1
p2 = subprocess.Popen(['python', 'emotions.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
import msvcrt
while(1):
    if msvcrt.kbhit():
        if ord(msvcrt.getch()) == 114: #this is r
            print("r")
            p2.stdin.write("Stop")
            p2.stdin.flush()

            
        if ord(msvcrt.getch()) == 119:
            print("w")
            p2.stdin.write("Run")
            p2.stdin.flush()
