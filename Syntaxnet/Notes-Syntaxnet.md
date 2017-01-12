
## Modified demo.sh (demo_carl.sh): <br>
Purpose of modification: <br>
1) Input from a text file <br>
2) Output to another text file <br>
`syntaxnet/demo_carl.sh --input=MAIN-IN >> output.txt` <br><br>

in demo_carl.sh…
- Update the input format variable from INPUT_FORMAT=stdin to INPUT_FORMAT=MAIN-IN <br>
From -> `[[ "$1" == "--conll" ]] && INPUT_FORMAT=stdin-conll || INPUT_FORMAT=stdin` <br>
To -> `[[ "$1" == "--conll" ]] && INPUT_FORMAT=stdin-conll || INPUT_FORMAT=MAIN-IN` <br><br>

in context.pbtxt (the one in parsey mcparseface folder)…
- Added the following lines of code to go fetch the input txt file... <br>
`input {` <br>
` name: "MAIN-IN"` <br>
` record_format: "english-text"` <br>
` Part { file_pattern:"carl/test.txt"}` <br>
`}`<br>

### [update] Wait!  There's an easier way...
You can ignore the above modifications and simply use the following command: <br>
`syntaxnet/demo.sh < input.txt > output.txt` <br>
