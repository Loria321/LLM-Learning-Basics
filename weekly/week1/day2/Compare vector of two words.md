 # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance")#创建pairwise_embedding_distance评估器实例
    words = ("apple", "iphone")#定义要比较的两个单词
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    #调用evaluate_string_pairs方法比较两个单词的嵌入向量距离
    print(f"Comparing ({words[0]}, {words[1]}): {x}")#打印比较结果
    #例如:Comparing (apple, iphone): 0.5678  # 实际输出是一个浮点数(0-1之间的距离,值越小表示两个单词的嵌入向量越接近)
