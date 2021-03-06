import xml.etree.ElementTree as ET
import pandas

ANSWER_LABELS_MAPPING = {'False': 0, 'True': 1, 'NonFactual': 2}
QUESTION_LABELS_MAPPING = {'Opinion': 0, 'Factual': 1, 'Socializing': 2}



def get_label(original_label, label_mapping):
    if original_label in label_mapping.keys():
        return label_mapping[original_label]
    return -1


def read_answer_labels_from_xml(input_xml_file):
    results = []
    print('parsing...', input_xml_file)

    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        question_tag = thread[0]
        question_fact_label = question_tag.attrib['RELQ_FACT_LABEL']
        if question_fact_label == 'Factual':
            for index, answer_tag in enumerate(thread):
                if index ==0 :
                    question_body = answer_tag[1].text
                    if(question_body==None):
                        question_body="none"
                if index > 0: # the 0 index was processed above - it is the question
                    answer_fact_label = answer_tag.attrib['RELC_FACT_LABEL']
                    answer_id = answer_tag.attrib['RELC_ID']
                    answer_body = answer_tag[0].text
                    if(answer_body==None):
                        answer_body="none"
                    label = get_label(answer_fact_label, ANSWER_LABELS_MAPPING)
                    if label > -1:
                        results.append([question_body,answer_body,label])
    return results

# Reads answer labels from file in the task input format.
def read_question_labels_from_xml(input_xml_file):
    labels = []
    print('parsing...', input_xml_file)
    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        question_tag = thread[0]
        question_id = question_tag.attrib['RELQ_ID']
        question_fact_label = question_tag.attrib['RELQ_FACT_LABEL']
        question_body = thread[0][1].text
        label = get_label(question_fact_label, QUESTION_LABELS_MAPPING)
        item = []
        item.append(question_id)
        item.append(question_body)
        item.append(label)
        labels.append(item)
    return labels




def get_no_id_labels(labels):
    no_id_labels = map(lambda item:list(filter(lambda x:item.index(x)!=0,item)),labels)
    return list(no_id_labels)


result = read_answer_labels_from_xml("answers_train.xml")

print len(result)


pandas_data = pandas.DataFrame(result,columns=["question","answer","label"])

pandas_data.to_csv("answer_train.tsv",sep="\t",header=True,index=False,encoding="utf-8")

print(pandas_data)