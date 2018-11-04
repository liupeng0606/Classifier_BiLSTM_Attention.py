import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
QUESTION_LABELS_MAPPING = {'Opinion': 0, 'Factual': 1, 'Socializing': 2}
body_lens=[]
def read_question_labels_from_xml(input_xml_file):
    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        q_id = thread[0].attrib["RELQ_ID"]
        q_subject = thread[0][0].text
        body = thread[0][1].text
        lable = QUESTION_LABELS_MAPPING[thread[0].attrib["RELQ_FACT_LABEL"]]
        if(body==None):
            body=""
        body_lens.append(len(body.split(" ")))
    nums = pd.Series(body_lens)
    print nums
    nums.plot(kind='bar')
    plt.show()
read_question_labels_from_xml("input_questions.xml")
