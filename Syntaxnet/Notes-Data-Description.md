# Notes on the data being parsed

Goal:  extract party names from .txt agreement files.

**Outstanding questions:** <br>
1) Is it safe to process only the first few lines of an agreement to be able to extract both party names? <br>
2) How to identify the end of a sentence when a period "." is used for multiple purposes such as "INC."? <br>

**Key observations:** <br>
1) Key words:  agreement, between, by (when close to "agreement" and "between"), and (when close to "agreement" and "between") <br>

<br>
**A. file name: Google_Construction_Agreement.txt** 
- Both party names are located in the first sentence of the text file
- Challenge:  periods "." are used for abbreviations in the text such as "INC."  So the logic cannot rely on periods to signify the end of a sentence.
- **Party1** is Google.  The word "between" precedes **Party1**.
- **Party2** is currenlty a blank line (I guess this blank line needs to be filled prior to processing...this makes sense).  The word "and" precedes **Party2**.

<br>
**B. file name: 2Think1 Solutions Inc_eula.txt**
- There is an end-user license agreement for "GIELTSHELP.COM" and "ACADEMICENGLISHHELP.COM" (*but this is not what we are looking for*)
- Challenge:  periods "." are used in website links and do not signify the end of a sentence.  So the logic cannot rely on periods to signify the end of a sentence.
- then, there is a legal agreement between "you" and "2THINK1 SOLUTIONS INC"
- **Party1** is you.  The word "between" precedes **Party1**.
- **Party2** is 2THINK1 SOLUTIONS INC.  The word "and" precedes **Party2** <br>
*note:*  code will need to ignore/skip the verbiage for GIELTSHELP.COM" and "ACADEMICENGLISHHELP.COM" and retrieve the information for **Party1** "you" and **Party2** "2THINK1 SOLUTIONS INC." 

<br>
**C. file name: BIZCOM_Master Services Agreement.txt**
- Both party names are are located in the first sentence of the text file
- Challenge:  periods "." are used for more than indicating the end of a sentence such as "bizcomweb.com"
- **Party1** is Bizcom Web Services. The word "between" precedes **Party1**.
- **Party2** is client.  The words "and the" precedes **Party2**.





