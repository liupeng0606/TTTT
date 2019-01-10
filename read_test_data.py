import xml.etree.ElementTree as ET
import pandas

ANSWER_LABELS_MAPPING = {'False': 0, 'True': 1, 'NonFactual': 2}
QUESTION_LABELS_MAPPING = {'Opinion': 0, 'Factual': 1, 'Socializing': 2}



def get_label(original_label, label_mapping):
    if original_label in label_mapping.keys():
        return label_mapping[original_label]
    return -1


# Reads answer labels from file in the task input format.
def read_answer_data_from_xml(input_xml_file):
    results = []
    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        question_body=''
        for item in thread:
            if((item.tag)=='RelQuestion'):
                question_body = item[1].text
            else:
                answer_body = item[0].text
                results.append([question_body,answer_body])
    return results


def read_answer_id_from_xml(input_xml_file):
    results = []
    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        for item in thread:
            if((item.tag)=='RelQuestion'):
                pass
            else:
                answer_id = item.attrib['RELC_ID']
                print (item.attrib['RELC_ID'])
                results.append(answer_id)
    return results





def read_question_data_from_xml(input_xml_file):
    questions_ids=[]
    questions_bodys=[]
    total = []
    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        question_tag = thread[0]
        question_id = question_tag.attrib['RELQ_ID']
        questions_ids.append(question_id)
        question_body = thread[0][1].text
        questions_bodys.append(question_body)
    total.append(questions_ids)
    total.append(questions_bodys)
    return total


def read_question_id_from_xml(input_xml_file):
    total = []
    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        question_tag = thread[0]
        question_id = question_tag.attrib['RELQ_ID']
        total.append(question_id)

    return total




result = read_answer_id_from_xml("answers_dev.xml")


pandas_data = pandas.DataFrame(result)

print(len(pandas_data))

pandas_data = pandas.DataFrame(result)

pandas_data.to_csv("answers_id.tsv",sep="\t",header=False,index=False,encoding="utf-8")

