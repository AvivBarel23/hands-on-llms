import os
import inspect
import datetime
import random

import openai
import time
from typing import List, Optional
from bytewax.outputs import DynamicOutput, StatelessSink
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff
from qdrant_client.models import PointStruct
from qdrant_client.conversions.common_types import ScoredPoint

from streaming_pipeline import constants
from streaming_pipeline.models import Document

# -- Global path to log in the same directory as this file
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "debug.log")

def debug_print(msg: str):
    """
    Logs debug messages to `debug.log` in this directory,
    including timestamp, filename, and line number.
    """
    # Capture call frame info (who called debug_print)
    frame_info = inspect.stack()[1]
    filename = os.path.basename(frame_info.filename)
    lineno = frame_info.lineno

    # Optional timestamp
    now_str = datetime.datetime.now().isoformat()

    formatted_msg = f"[{now_str}][{filename}:{lineno}] {msg}"
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")


def build_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
    """
    Builds a QdrantClient object with the given URL and API key.

    Args:
        url (Optional[str]): The URL of the Qdrant server. If not provided,
            it will be read from the QDRANT_URL environment variable.
        api_key (Optional[str]): The API key to use for authentication. If not provided,
            it will be read from the QDRANT_API_KEY environment variable.

    Raises:
        KeyError: If the QDRANT_URL or QDRANT_API_KEY environment variables are not set
            and no values are provided as arguments.

    Returns:
        QdrantClient: A QdrantClient object connected to the specified Qdrant server.
    """
    debug_print("[DEBUG] build_qdrant_client START")

    if url is None:
        try:
            url = os.environ["QDRANT_URL"]
        except KeyError:
            raise KeyError(
                "QDRANT_URL must be set as environment variable or manually passed as an argument."
            )

    if api_key is None:
        try:
            api_key = os.environ["QDRANT_API_KEY"]
        except KeyError:
            raise KeyError(
                "QDRANT_API_KEY must be set as environment variable or manually passed as an argument."
            )

    client = QdrantClient(url, api_key=api_key)
    debug_print("[DEBUG] QdrantClient built successfully")

    debug_print("[DEBUG] build_qdrant_client END")
    return client


class HierarchicalDataManager:
    def __init__(self, qdrant_client: QdrantClient,collection_name):
        self.new_node_id=0
        debug_print("[DEBUG] HierarchicalDataManager.__init__ START")
        self.client = qdrant_client
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.indices_collection:str = collection_name

    def classify_with_gpt(self, text: str, options: List[str], level: str) -> str:
        max_retries = 5  # Maximum number of retry attempts
        retry_delay = 2  # Initial delay between retries (in seconds)
        attempt = 0

        while attempt < max_retries:
            try:
                debug_print("[DEBUG] classify_with_gpt START")
                debug_print(f"[DEBUG] text='{text[:50]}...' options={options} level={level}")

                prompt = (
                    f"Based on the following text, decide which {level} it belongs to:\n\n"
                    f"Text: {text}\n\n"
                    f"Options: {', '.join(options)}\n\n"
                    f"Only return the name of the {level}. If there is no correct option, please suggest one."
                )
                #debug_print(prompt)

                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a financial classifier for data"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.8,
                    max_tokens=10,
                    top_p=1
                )

                debug_print("[DEBUG] classify_with_gpt before ")
                classification = response.choices[0].message.content.strip().replace(".", "")
                debug_print("[DEBUG] classify_with_gpt after")
                debug_print(f"[DEBUG] GPT classification result: {classification}")

                if classification not in options:
                    debug_print(
                        f"[DEBUG] classification '{classification}' not in existing options => treating as new"
                    )
                    debug_print("[DEBUG] classify_with_gpt END (NEW LABEL)")
                    return classification

                debug_print("[DEBUG] classify_with_gpt END (EXISTING LABEL)")
                return classification

            except Exception as e:
                attempt += 1
                wait_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                debug_print(f"Exception :{e}, [DEBUG] Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt}/{max_retries})")
                time.sleep(wait_time)


        # If all retries fail, return an appropriate error message or raise an exception
        debug_print("[DEBUG] classify_with_gpt END (FAILED AFTER RETRIES)")
        raise Exception("Rate limit exceeded. Maximum retries reached.")


    def get_hierarchy_node(self, name: str, level: str) -> Optional[ScoredPoint]:
        debug_print("[DEBUG] get_hierarchy_node START")
        debug_print(f"[DEBUG] name='{name}', level='{level}'")

        results = self.client.search(
            collection_name=self.indices_collection,
            query_vector=[1],  # Dummy query vector
            filter={
                "must": [
                    {"key": "name", "match": {"value": name}},
                    {"key": "type", "match": {"value": level}},
                ]
            },
            limit=1,
        )
        if results:
            debug_print("[DEBUG] Found matching node(s). Returning the first one.")
        else:
            debug_print("[DEBUG] No matching node found.")

        debug_print("[DEBUG] get_hierarchy_node END")
        return results[0] if results else None


    def save_hierarchy_node(self,
                            name: str,
                            level: str,
                            parent: Optional[str] = None,
                            children: Optional[List[str]] = None):
        debug_print("[DEBUG] save_hierarchy_node START")
        debug_print(
            f"[DEBUG] name='{name}', level='{level}', parent='{parent}', children={children}"
        )

        node = self.get_hierarchy_node(name, level)
        if node:
            debug_print("[DEBUG] Node exists; updating existing node.")
            payload = node.payload
            payload["children"] = list(set(payload.get("children", []) + (children or [])))
            self.client.upsert(
                collection_name=self.indices_collection,
                points=[
                    PointStruct(
                        id=node.id,  # use the actual scored point ID
                        vector=[1.0] * 384,
                        payload=payload,
                    )
                ],
            )
        else:
            debug_print("[DEBUG] Node does not exist; creating new node.")
            try:
                self.new_node_id +=1
                self.client.upsert(
                    collection_name=self.indices_collection,
                    points=[
                        PointStruct(
                            id=self.new_node_id,
                            vector=[1.0] * 384,  # Dummy vector
                            payload={
                                "type": level,
                                "name": name,
                                "parent": parent,
                                "children": children or [],
                            },
                        )
                    ],
                )
            except Exception as e:
                debug_print(f"[DEBUG] exception :{e} , tried to insert node to collection {self.indices_collection}  ")


        debug_print("[DEBUG] save_hierarchy_node END")


    def save_data(self, document):
        debug_print("[DEBUG] save_data START")
        document_text = ' '.join(document.text)
        debug_print("[DEBUG] Full document text: " + document_text[:100] + "...")
        try:
            # # Step 1: Sector Classification
            # cleaned_question = self.clean(question_str)
            # # pass them through the model and average the embeddings.
            # cleaned_question = cleaned_question[: self.embedding_model.max_input_length]
            # embeddings = self.embedding_model(cleaned_question)
            #
            # # (or other time frame).
            # matches = self.vector_store.search(
            #     query_vector=embeddings,
            #     k=self.top_k,
            #     collection_name=self.vector_collection,
            # )
            #
            empty_vector = [[0.14927152962651125, 0.8221775477249196, 0.7674495687373497, 0.2222706599579336, -0.6740345735067472, 0.07272692017042748, -0.4562144397516006, 0.08568619549162126, -0.17121403750649544, 0.956817623790791, 0.940629898125686, -0.7522964818107729, 0.9712934592215516, -0.25408020373545015, 0.5026104233210458, -0.5435373779778685, -0.9815792736398925, 0.745086653317532, -0.05531377557344075, -0.8026411015922563, 0.9722916685754934, -0.6183810645415688, -0.3865082945420226, 0.0015416382040411847, 0.8603290577284328, 0.4150713147435725, -0.0783134322512653, -0.754804368707273, -0.2788880546600323, 0.05748915594485893, 0.7247862289237408, -0.24642851295165524, -0.7703451549316342, -0.9135680717632064, -0.7587328479961446, 0.5244375540667172, -0.5793230649954819, -0.7461698601035573, -0.6011001040345292, 0.5362302807674062, 0.45926174416560106, -0.5935729232520579, 0.3402721745495587, -0.7453392336824165, -0.7478721184570698, 0.5965218687969502, -0.4009853161438448, 0.07086524145578887, -0.7454555914394299, 0.4184471669919543, 0.9574293395945237, -0.695432420557061, 0.9481485732226449, 0.6512957908716326, -0.9019324144244041, -0.6576340993991636, 0.8212754684098083, -0.18765998543806672, -0.7616558967505724, 0.3104758451298453, 0.24934134618754422, -0.8227472511369118, -0.9818102315058463, -0.2818843431801126, 0.48474630161499643, -0.025349030511584436, 0.4499249858795571, -0.580642891627638, -0.2415339353897743, 0.6490288734050518, 0.5862343291428123, -0.8313964314971558, 0.7828107714515558, 0.29736891961824785, -0.43359287533306223, 0.8607216007005223, 0.26051768817919174, 0.8516572618314517, 0.03828159318820701, 0.12130900995524829, -0.43303375749732353, 0.09111306650644901, -0.9085855875991871, -0.27136533705399146, 0.2991753886827353, -0.6898345792160676, 0.7235727972878101, -0.10294257642811022, -0.7651997408226665, -0.1024130829807306, 0.024612262073347102, -0.1709843848025001, 0.5042381736691779, -0.32162221732531693, -0.3681528214444989, -0.6674459171965514, 0.9124397521935395, -0.46944378630336425, 0.5155486913758431, 0.05333079375525185, -0.07322325146942155, 0.9935171891189227, 0.5126425322199384, 0.3812515726972048, 0.9985968547353978, 0.16325999048335937, 0.8869867450097237, 0.7440735253524711, -0.9417761974032293, 0.7972742700415902, -0.024115025288838687, -0.9601511708502148, 0.14702661309899567, -0.48035500712528734, 0.44194275916521186, -0.6257738859310811, 0.8893916652749463, 0.984544481350383, 0.04931428011355177, -0.5703081090485698, 0.5868711334861978, -0.2671305039246381, -0.32758343295057024, 0.9228163248275743, 0.2405385513092051, 0.9794967791319928, 0.6508009492203162, -0.2288478104455598, 0.14960051416173714, -0.18672388578765542, 0.7344402155039258, 0.5992647503290633, 0.3875088940519298, 0.5000321905447718, -0.45689765160552054, 0.4373958921940473, 0.7280609416341584, 0.9203569513988694, -0.46839157117132557, 0.37597511992880395, -0.5582014709861955, 0.8167673212491504, 0.676642825172926, -0.2819011154762232, 0.9070104715617142, -0.3151196178303646, -0.9515613522393227, 0.5488952418619486, 0.29510709529516155, 0.5480738237700793, -0.5086657130432397, 0.5023888183672256, -0.06388553283215925, 0.3874733577745153, 0.3567088032269019, -0.8690354535906315, -0.5679893885010008, 0.03545279616204766, -0.060218924109458394, 0.10089663834051699, 0.17102525094239085, -0.29976456947240226, -0.5230194300536644, 0.5192026687539122, -0.6218265933950187, -0.41140324378672855, 0.5819629120311727, 0.6207985129806315, -0.2995342170612292, 0.07271495511091475, -0.14998954637141493, 0.305119411351904, 0.7168689286740755, -0.37378948208578366, -0.43872003173345187, 0.6394906172160071, -0.9706883389465757, 0.3370064942064548, 0.8649099241451887, -0.8886722891402838, -0.5766747317597989, -0.7269167278904338, 0.9585736982444972, 0.28516127130581626, -0.5548600368212524, -0.9253940454114413, -0.923776284745957, 0.158633295866643, 0.6815015676628557, -0.3840152582034999, 0.6039753875363705, -0.8676595974602284, 0.6332491013924697, -0.841170508986002, -0.08503838808509068, 0.2812738042650471, 0.594613960834824, 0.6394282557186481, 0.3236246455024432, 0.18382716954976197, -0.33539723849005343, -0.8671263474703785, -0.7419320450399376, 0.1881350777212445, -0.3499878122985789, 0.8944907419305332, 0.869405443878569, 0.049576058368095755, 0.8409386843191189, -0.9358321799551226, 0.6108089331061386, 0.179883492964942, 0.46852952503499545, 0.5263456697225042, -0.581480732493433, 0.6554371009449729, -0.21186331824581317, 0.2979306455396453, -0.36617311022403, -0.21735251262594368, 0.053514069686602106, 0.24231377473735938, 0.11716362755544196, 0.04181117901909803, -0.2665725703560742, 0.5666671689475098, -0.10846931935359039, 0.045819268492941934, 0.818200346946546, -0.13358559525960811, -0.9561354549518088, 0.22189463238901075, -0.11648913952189477, 0.25078755897459004, 0.8051523339824114, 0.6016546073584488, -0.8582985301453256, 0.13684667624898816, -0.6224992302417724, -0.25824777619988604, -0.22356111144504132, -0.8513008129923882, 0.4181851943567041, 0.15274091175755866, 0.5833612039080509, 0.6016697390479275, -0.11392679801085892, -0.32592258619504877, -0.2469702888198695, -0.013537585933380392, 0.5433022790544833, 0.7289158521315222, -0.1603505714606266, 0.8061862342109953, 0.6577082155075153, 0.396673695415666, -0.5979472408805622, -0.7802800479315244, -0.9096199055611809, 0.15519211391896004, -0.2608901161183885, -0.4431030517128576, -0.049846163580973046, -0.5031745989940175, -0.42630430307027356, -0.7153717106890813, 0.17771032658786123, 0.11357307255443883, -0.8794615519572679, 0.08910524288948696, -0.7273039603228855, -0.8135621559096198, -0.6898388282252674, -0.20248763444406315, -0.9774613435645172, 0.37708903401849914, 0.17350184655668732, 0.445959374003734, 0.18528483994109957, -0.5573761891921525, -0.0739928547026587, -0.786391522237619, -0.884652913668601, -0.998180300065866, -0.3109907045490603, 0.3673725763284521, 0.0431724239197977, 0.583319348959983, -0.17785724038389872, -0.21268611782614366, -0.18158446693894414, -0.7539205423847888, 0.6390790100796484, 0.8605931746495923, 0.7880068299097693, 0.6120150168381859, -0.20639431029367117, -0.7179090923711633, -0.9701451857335672, -0.7893444693902631, -0.9840694123785729, 0.6658420136617107, -0.06579862781920709, 0.9281326178489953, -0.44776931796191755, 0.7172783001422707, -0.9803155183034002, -0.1755307670215267, 0.07704276319624626, 0.1972969674239109, 0.9996114922515715, -0.412341623123458, -0.5630852097165873, 0.7725328340456641, 0.24953074176608148, -0.7861834048339684, -0.9962454624844548, -0.6563572020604318, 0.7009252693908827, -0.09025613454869741, 0.3229022484330808, 0.7360464497846884, 0.8670308622725356, 0.4761133622536884, -0.5340775008144223, -0.8798411151893777, 0.2591313376783957, 0.07360138349687095, -0.7538286451189802, 0.5215989255848699, -0.42104829437088465, 0.9178515538764977, 0.11708618296534667, -0.5375309833006803, -0.8430093179958222, 0.007557242255352037, -0.713294773339612, -0.024360393864719887, -0.2743380193775147, 0.3792548571211518, -0.6445852263683773, 0.6847230620203877, -0.7914054947183216, -0.19546863552144633, 0.5649436336769842, -0.8436096642542497, 0.34341862578618, -0.3640643051211221, -0.8471824001786898, -0.15655545859977726, -0.42909582108026556, 0.6702889172587758, 0.01505729038047865, 0.5514817477915614, 0.3300409451716564, -0.7025140450863419, -0.5575926065784895, -0.11180415939470745, -0.2787952887178209, 0.2248563865357942, 0.9966291048992704, -0.46148203056927306, -0.35710405246234256, -0.08015081514115407, -0.29120040539929226, -0.02414473932853034, -0.059733348020237775, 0.7476685514365251, 0.7446638158907455, 0.7869839981466817, -0.5327030861354456, -0.05606395202989112, 0.09159739464063521, 0.17471390968378842, 0.07281874476344607, 0.10379376651548156, 0.5547330863108009, -0.7559655485558039, -0.6961066199779946, -0.262725963976135, 0.11026527798635843, -0.6158794902370583, -0.516562642293718, 0.7267110183623571]]
            sectors = [
                node.payload["name"] for node in self.client.search(
                    query_vector=empty_vector,
                    collection_name=self.indices_collection,
                    filter={"must": [{"key": "type", "match": {"value": "sector"}}]}
                )
            ]
        except Exception as e:
            debug_print("[DEBUG] exception!!!!!!!!!!!!!!!!!!!!!!!!!!!!:" + str(e))
            sectors=[]

        debug_print(f"[DEBUG] Found existing sectors: {sectors}")
        sector = self.classify_with_gpt(document_text, sectors, "sector")
        debug_print(f"[DEBUG] sector => '{sector}'")
        self.save_hierarchy_node(name=sector, level="sector")
        debug_print(f"[DEBUG] saved sector")


    # Step 2: Company/Subject Classification
        subjects_raw =  self.client.search(
                collection_name=self.indices_collection,
                query_vector=[1.0] * 384,
                filter={
                    "must": [
                        {"key": "type", "match": {"value": "subject"}},
                        {"key": "parent", "match": {"value": sector}},
                    ]
                }
            )
        debug_print(f"[DEBUG] subjects => '{subjects_raw}'")
        subjects =[ node.payload["name"] for node in subjects_raw]
        debug_print(f"[DEBUG] Found existing subjects under sector: {subjects}")
        subject = self.classify_with_gpt(document_text, subjects, "subject")
        debug_print(f"[DEBUG] subject => '{subject}'")
        self.save_hierarchy_node(name=subject, level="subject", parent=sector)
        debug_print(f"[DEBUG] saved subject")

        # Step 3: Event Type Classification
        event_types = [
            node.payload["name"] for node in self.client.search(
                collection_name=self.indices_collection,
                query_vector=[1.0] * 384,
                filter={
                    "must": [
                        {"key": "type", "match": {"value": "event_type"}},
                        {"key": "parent", "match": {"value": subject}},
                    ]
                }
            )
        ]
        debug_print(f"[DEBUG] Found existing event_types under subject: {event_types}")
        event_type = self.classify_with_gpt(document_text, event_types, "event type")
        debug_print(f"[DEBUG] event_type => '{event_type}'")
        self.save_hierarchy_node(name=event_type, level="event_type", parent=subject)

        debug_print(f"[DEBUG] saved event_type: {event_type}")


        # Step 4: Save the document in its specific Qdrant collection
        collection_name = f"alpaca_financial_news_{sector}_{subject}_{event_type}".lower().replace(" ", "_")
        debug_print(f"[DEBUG] Final collection_name => '{collection_name}'")
        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            debug_print(f"[DEBUG] Collection '{collection_name}' does NOT exist; creating.")
            if not document.embeddings:
              raise ValueError("document.embeddings is missing or empty.")
            vector_size = len(document.embeddings[0])
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            debug_print("[DEBUG] Created new collection with vector_size=" + str(vector_size))


        debug_print("[DEBUG] Upserting the document's embeddings...")
        ids, payloads = document.to_payloads()
        points = [
            PointStruct(id=idx, vector=vector, payload=_payload)
            for idx, vector, _payload in zip(ids, document.embeddings, payloads)
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        debug_print(f"[DEBUG] Document saved successfully in {collection_name}")

        debug_print("[DEBUG] save_data END")


class QdrantVectorSink(StatelessSink):
    """
    A sink that writes document embeddings to a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client to use for writing.
        collection_name (str, optional): The name of the collection to write to.
            Defaults to constants.VECTOR_DB_OUTPUT_COLLECTION_NAME.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):
        debug_print("[DEBUG] QdrantVectorSink.__init__ START")
        self.hierarchical_data_manager = HierarchicalDataManager(client,collection_name)
        debug_print("[DEBUG] QdrantVectorSink.__init__ END")

    def write(self, document: Document):
        debug_print("[DEBUG] QdrantVectorSink.write START")
        self.hierarchical_data_manager.save_data(document)
        debug_print("[DEBUG] Document saved to hierarchical data store!")
        debug_print("[DEBUG] QdrantVectorSink.write END")


class QdrantVectorOutput(DynamicOutput):
    """A class representing a Qdrant vector output.

    This class is used to create a Qdrant vector output, which is a type of dynamic output that supports
    at-least-once processing. Messages from the resume epoch will be duplicated right after resume.

    Args:
        vector_size (int): The size of the vector.
        collection_name (str, optional): The name of the collection.
            Defaults to constants.VECTOR_DB_OUTPUT_COLLECTION_NAME.
        client (Optional[QdrantClient], optional): The Qdrant client. Defaults to None.
    """

    def __init__(
        self,
        vector_size: int,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        client: Optional[QdrantClient] = None,
    ):
        debug_print("[DEBUG] QdrantVectorOutput.__init__ START")
        self._collection_name = collection_name
        self._vector_size = vector_size

        if client:
            self.client = client
        else:
            self.client = build_qdrant_client()
        try:
            self.client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, ValueError):
            self.client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
                # Manuall add this optimizers_config to address issue: https://github.com/iusztinpaul/hands-on-llms/issues/72
                # qdrant_client.http.exceptions.ResponseHandlingException: 1 validation error for ParsingModel[InlineResponse2005] (for parse_as_type)
                # obj -> result -> config -> optimizer_config -> max_optimization_threads
                # none is not an allowed value (type=type_error.none.not_allowed)
                optimizers_config=OptimizersConfigDiff(max_optimization_threads=1),
            )

    def build(self, worker_index, worker_count):
        """Builds a QdrantVectorSink object.

        Args:
            worker_index (int): The index of the worker.
            worker_count (int): The total number of workers.

        Returns:
            QdrantVectorSink: A QdrantVectorSink object.
        """

        debug_print(f"[DEBUG] QdrantVectorOutput.build START on worker {worker_index}/{worker_count}")
        sink = QdrantVectorSink(self.client, self._collection_name)
        debug_print("[DEBUG] QdrantVectorOutput.build END - returning sink")
        return sink
