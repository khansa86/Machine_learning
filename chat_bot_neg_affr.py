
import aiml

# Create the kernel and learn AIML files
kernel = aiml.Kernel()
kernel.learn("assignment.aiml")


while True:
    print (kernel.respond(input("Enter your message >> ")))
