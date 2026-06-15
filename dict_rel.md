list_relations  = {
"FormOf" : "[SUBJECT] is an inflected form of [OBJECT]" ,
"IsA" : "[SUBJECT] is a strict taxonomic subtype, class, or specific instance of [OBJECT]" ,
"CouldBe" : "[SUBJECT] is conditionally, hypothetically, or possibly described as [OBJECT], categorized as [OBJECT], or executing [OBJECT]",
"PartOf": "[SUBJECT] is an intrinsic physical component, member, or structural part of [OBJECT]",
"HasA" : "[SUBJECT] physically possesses or owns [OBJECT]",
"Contains" : "[SUBJECT] physically encloses, holds inside, or is chemically/structurally composed of [OBJECT] inside it",
"HasQuantity" : "[SUBJECT] has a quantity of [OBJECT]",
"UsedFor" : "[SUBJECT] is used for [OBJECT]; the purpose of [SUBJECT] is [OBJECT]" ,
"CapableOf" : "Something that [SUBJECT] can typically do is [OBJECT]" ,
"AtLocation" : "[SUBJECT] is located at [OBJECT], [SUBJECT] can be an event taking place at [OBJECT]" ,
"AtTime" : "[SUBJECT] took place in, during, or relative to the temporal frame, date, or event [OBJECT]",
"Causes" : "[SUBJECT] and [OBJECT] are events, and [SUBJECT] causes [OBJECT]" ,
"HasSubevent" : "[SUBJECT] and [OBJECT] are events, and [OBJECT] happens as a subevent of [SUBJECT]" ,
"HasPrerequisite" : "In order for [SUBJECT] to happen, [OBJECT] needs to happen; [OBJECT] is a dependency of [SUBJECT]" ,
"HasProperty" : "[SUBJECT] has [OBJECT] as a property; [SUBJECT] can be described as [OBJECT]" ,
"MotivatedByGoal" : "Someone does [SUBJECT] because they want result [OBJECT]; [SUBJECT] is a step toward accomplishing the goal [OBJECT]" ,
"CreatedBy" : "[OBJECT] is a process or agent that creates [SUBJECT]" ,
"Synonym" : "[SUBJECT] and [OBJECT] have very similar meanings. Symmetric" ,
"Antonym" : "[SUBJECT] and [OBJECT] are opposites in some relevant way" ,
"HasContext" : "[SUBJECT] is inherently associated with, defined by, or belongs to the domain, or category [OBJECT]" ,
"SimilarTo" : "[SUBJECT] is similar to [OBJECT]. Symmetric" ,
"MadeOf" : "[SUBJECT] is made of [OBJECT]" ,
"ReceivesAction" : "[OBJECT] can be done to [SUBJECT]",
"PerformsAction" : "[SUBJECT] is doing the action [OBJECT], [OBJECT] is usually a verb"}


prompt = f"""
Extract all factual ([SUBJECT], relation, [OBJECT]) triples from sentence. 
One triple per line in the format: 
[SUBJECT] | relation | [OBJECT]

No explanations. If no triple can be extracted, write nothing. 

Allowed Relations and Type Constraints:
{json.dumps(list_relations, indent=2)}
Do not use any other relation.

EXEMPLES:
Input: John is eating a ham sandwich
Output:
John | PerformsAction | eating
sandwich | ReceivesAction | eating
sandwich | Contains | ham

Input: A man is playing an instrument
Output:
man | PerformsAction | playing
instrument | ReceivesAction | playing

Input: Simon didn't call me back. He must be busy.
Output:
Simon | PerformsAction | not call
me | ReceivesAction | not call
Simon | CanBe | busy

Sentence: {text}"""