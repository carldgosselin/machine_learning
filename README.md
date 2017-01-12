# Syntaxnet
Sharing Syntaxnet experiences

# Installation on OSX
I followed the installation instructions in the following links:
- https://bazel.build/versions/master/docs/install.html#mac-os-x
- https://github.com/tensorflow/models/tree/master/syntaxnet

# Personal experience during installation

**I already had a few components installed on my machine** <br>
1) Homebrew already installed. I used Homebrew to install the rest of the components for Syntaxnet <br>
2) I already had Python 2.7.10 installed on my machine <br>
3) I already had pip 9.0.1 installed on my machine <br>

**Bazel installation (and re-installation)** <br>
*note:  I installed the wrong version of bazel (0.4.0).  Had to uninstall and re-install bazel 0.3.1* <br>
4) Installed bazel -> `brew install bazel` <br>
5) Upgraded bazel -> `brew upgrade bazel` (upgarded to wrong version, 0.4.0, had to uninstall)    
6) Uninstall bazel -> `brew uninstall bazel` <br>
7) Search and downloand bazel 0.3.1 at https://github.com/bazelbuild/bazel/releases <br>
*note: I downloaded bazel-0.3.1-installer-darwin-x86_64.sh* <br>
8) Execute bazel-0.3.1-installer-darwin-x86_64.sh in command prompt -> `./bazel-0.3.2-installer-darwin-x86_64.sh` <br>
9) Verify bazel version -> `bazel version` <br>
result: `Build label: 0.3.1`

**Syntaxnet (built by TensorFlow) folders and files installation** <br>
10) clone the syntaxnet from github -> `git clone --recursive https://github.com/tensorflow/models.git` <br>
11) navigate to the folder where the configuration file resides -> `cd models/syntaxnet/tensorflow` <br>
12) Execute configuration file -> `./configure` <br>
*note:  Multiple questions will be asked during installation.  I kept all defaults by pressing `Enter` and chose `N` for every [Y/N] question (Google Cloud support?, Hadoop support?, etc..)* <br>

**Build and test bazel** <br>
13) Build and test bazel -> `bazel test syntaxnet/... util/utf8/...`<br>
*note:  the installation recommends to run the following on a mac:* <br>
`bazel test --linkopt=-headerpad_max_install_names syntaxnet/... util/utf8/...` <br>
*However, I chose to run the code in number 13.* <br>

* result: after 30 minutes or so, bazel completed the build and test with all tests passed (although with a lot of warning messages about unused code/variables). <br>

**Final test** <br>
14) I parsed the following text to see how SyntaxNet worked... <br>
`echo 'Bob brought the pizza to Alice' | syntaxnet/demo.sh` <br>
* result: <br>
... <br>
Input: Bob brought the pizza to Alice <br>
Parse: <br>
brought VBD ROOT <br>
 +-- Bob NNP nsubj <br>
 +-- pizza NN dobj <br>
 |   +-- the DT det <br>
 +-- to IN prep <br>
     +-- Alice NNP pobj <br>




