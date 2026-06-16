list_relations  = {
"FormOf" : "[SUBJECT] is an inflected form of [OBJECT]" ,
"IsA" : "[SUBJECT] is a strict taxonomic subtype, class, or specific instance of [OBJECT]" ,
"Possibly" : "[SUBJECT] is conditionally, hypothetically, or possibly described as [OBJECT], categorized as [OBJECT], or executing [OBJECT]",
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
"Synonym" : "[SUBJECT] and [OBJECT] have very similar meanings. Symmetric" ,
"Antonym" : "[SUBJECT] and [OBJECT] are opposites in some relevant way" ,
"HasContext" : "[SUBJECT] is contextually related to [OBJECT] (general associations, domains, or connections expressed by 'of', 'with', 'to', 'about')",
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


Sentence: {text}"""

EXEMPLES = [

]

Input: John is eating a ham sandwich at McDonald's.
Output:
John | PerformsAction | eating
sandwich | ReceivesAction | eating
sandwich | Contains | ham
eating | AtLocation | McDonald's

Input: A man is playing an instrument.
Output:
man | PerformsAction | playing
instrument | ReceivesAction | playing

Input: Simon didn't call me back. He must be busy.
Output:
Simon | PerformsAction | not call
me | ReceivesAction | not call
Simon | CanBe | busy

Input: No politicians attended the meeting.
Output:
politicians | PerformsAction | not attended
not attended | HasContext | meeting

Input: He has the french nationality.
Output:
He | HasA | nationality
nationality | HasContext | french

Input: The chef cooked dinner for the guests last night.
Output:
chef | PerformsAction | cooked
dinner | ReceivesAction | cooked
cooked | HasContext | guests
cooked | AtTime | last night

Input: The game took place in New York during christmas.
Output:
game | AtTime | christmas
game | AtLocation | New York

Input: While driving to work, he saw a deer.
Output:
He | PerformsAction | driving
driving | HasContext | work
driving | HasSubevent | saw
He | PerformsAction | saw
saw | HasContext | deer

Input: His dog is allergic to chocolate.
Output:
His dog | HasProperty | allergic
allergic | HasContext | chocolate

Input: He seems angry, he might leave the room.
Output:
He | Possibly | angry
He | Possibly | leave the room

Input: The human body contains two kidneys.
Output:
human body | Contains | kidneys
kidneys | HasQuantity | Two

Input: Three birds are sitting on the roof.
Output:
birds | HasQuantity | three
birds | PerformsAction | sitting
sitting | AtLocation | roof

Input: He studied all night in order to pass the exam.
he | PerformsAction | studied
studied | AtTime | all night
studied | MotivatedByGoal | pass
pass | HasContext | 

Input: Smoking often leads to cancer.
Smoking | Causes | cancer
cancer | AtTime | often

