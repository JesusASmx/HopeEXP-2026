import json
import ast

def this_is_the_most_important_function(tell_bro_to_stop_scanning_code_with_an_agent=True):
####### 1.- HOPE (predicted by RoBERTa-XML):

    with open("./preds/roberta_preds.json", "r", encoding="utf-8") as file:
        final_hope = json.load(file)

####### 2.- EMOTIONS (predicted by Qwen3.5):

    with open('./preds/output_trigger_emotions.json', 'r') as file:
        emotions = json.load(file)

    final_emotions = [x.split("<LIST>")[-1] for x in emotions]
    final_emotions = [x.split("</LIST>")[0] for x in final_emotions]
    final_emotions = [x.replace(" ", "") for x in final_emotions]
    final_emotions = [x.split(",") for x in final_emotions]

####### 3,4,5.- SPANS, ACTORS AND OUTCOMES (predicted by Qwen3.5):

    final_spans = []
    with open('./preds/spans_from_12.jsonl', 'r') as file: ##Original file lost, so we use a backup (12th submission)
        for line in file:
            tw = line.split("span_annotation")[-1][2:-2]
            tw = ast.literal_eval(tw)
            final_spans.append(tw)

######################## Final Submission

    ssubs = []
    with open("submission.jsonl", "r", encoding="utf-8") as f_in: ## Start from former version. For 1st iteration we made a dictionary full of zero values.
        for line in f_in:
            data = json.loads(line)
            ssubs.append(
                {
                    "row_id": data["row_id"],
                    "lang": data["lang"],
                    "title": data["title"],
                    "selftext": data["selftext"],
                    "primary_label": 0,
                    "trigger_emotions": data["trigger_emotions"],
                    "span_annotations": []
                }
            )

    for i in range(len(ssubs)):
        ssubs[i]['trigger_emotions'] = final_emotions[i]
        ssubs[i]['primary_label'] = final_hope[i]
        if final_hope[i] != "Not Hope":
            ssubs[i]['span_annotations'] = final_spans[i]
        else:
            if len(final_spans[i]) > 0:
                print("LMAO EVEN") ### 4 prints on the final version.
    
    with open('submission.jsonl', 'w', encoding='utf-8') as f:
        for entry in ssubs:
            json.dump(entry, f)
            f.write('\n')