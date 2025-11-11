# Create a script that detects and logs unusually large functions/classes for easier chuncking after
import ast
import os

#Reading through all the lines of a file to get a idea of how amny lines there are present
def read_lines_of_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        line_count = 0
        for lines in file:
            line_count = line_count + 1
    return line_count
#Function to detect large blocks of code
def detect_large_blocks(filename):
    with open(filename, "r", encoding="utf-8") as file: # Reading the file
        max_lines = 45 # Max line count equals 100 can't exceed it
        file_content = file.read() # reading the contents of a file
        tree = ast.parse(file_content) # I converted the code in AST so that python understands what's going on as a data structure

        for node in ast.walk(tree): # Loops through each node inside the tree
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)): #Checks if the current function represents a function or a class
                start = node.lineno # number of line where the function/class starts
                end = getattr(node, "end_lineno", start) # number of line where the function/class ends
                length = end - start # Determing the file length

                if length > max_lines: #Checking condition if the length is greater then max lines
                    print("Large Block detected: " + str(node.name) + " is " + str(length) + " lines long!") # then print Large block detected

    

#Main function for testing and debugging

def main():
    #test file path will work on directly with file names as well
    test_file = r"C:\Users\aryan\OneDrive\Kingdom Rush\Documents\SecureByte\SecureBYTE_AI-1\examples\multi_provider_comparison.py"
    print("Current working directory:", os.getcwd())
    try:
        detect_large_blocks(test_file)
    except FileNotFoundError:
        print("the file was not found, please recheck and verify!")

if __name__== "__main__":
    main()
