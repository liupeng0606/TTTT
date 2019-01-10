import pandas



def get_task_a():


    total=[]

    data_id_list = pandas.read_csv("questions_id.tsv").values.tolist()

    data_id_list = map(lambda x:x[0],data_id_list)


    data_result_list = pandas.read_csv("hell.txt").values.tolist()

    index_list = list(map(lambda x: x.index(max(x)),data_result_list))

    for i in range(len(index_list)):
        total.append([data_id_list[i],index_list[i]])

    data = pandas.DataFrame(total)

    data.to_csv("question_final.tsv", sep="\t", header=False, index=False, encoding="utf-8")


    pass



def get_task_b():

    total=[]

    data_id_list = pandas.read_csv("answers_id.tsv").values.tolist()

    data_id_list = map(lambda x:x[0],data_id_list)

    print data_id_list


    data_result_list = pandas.read_csv("answer_test_result.tsv",sep="\t").values.tolist()

    index_list = list(map(lambda x: [x.index(max(x)),max(x)],data_result_list))

    for i in range(len(index_list)):
        total.append([data_id_list[i], index_list[i][0],  index_list[i][1]])

    data = pandas.DataFrame(total)

    data.to_csv("answer_final.tsv", sep="\t", header=False, index=False, encoding="utf-8")



    pass



get_task_a()


