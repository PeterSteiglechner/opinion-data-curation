# Opinion Data – The collective minds kinda way

We are collecting opinion data on multiple data sets


The variables should include all of the following

* source
* country
* time/year of survey
* metainfo (string containing the wave number, panel, ...)
* attention_to_politics_general (*)
* most_important_topic
* second_most_important_topic
* opinions on different topics, variables coded as f"econ_{actual_variable_name}" for categories – econ, climate, civil (gay,abortion...), migr, ...

If something is not tracked in a specific dataset, it should be coded as NaN


Datasets included:
- ANES
- ESS
- GESIS
- AUTNET
