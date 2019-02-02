import random, string, re   
import xml.etree.ElementTree as ET

idStartVocab = string.ascii_letters + "_"
idVocab = string.digits + idStartVocab + "."
textVocab = re.sub("[\x0c\x0b.<>/]", "", string.printable)

def getId(len_range):
    return (
        random.choice(idStartVocab)
        + "".join([
            random.choice(idVocab) for _ in range(*len_range)
        ])
    )

def getText(len_range, vocab=None):
    textLen = random.randrange(*len_range)
    if textLen < 0:
        return None

    if vocab is None:
        vocab = textVocab
    return "".join([random.choice(textVocab) for _ in range(textLen)])

def generateVocabs(generatorArgs):
    tag_gen_params = generatorArgs.get("tag_gen_params", (50, (0, 5)))
    attr_gen_params = generatorArgs.get("attr_gen_params", (50, (0, 5)))
    attr_value_gen_params = generatorArgs.get("attr_value_gen_params", (50, (0, 20)))

    tagSet = [getId(tag_gen_params[1]) for _ in range(tag_gen_params[0])]
    attrSet = [getId(attr_gen_params[1]) for _ in range(attr_gen_params[0])]
    attrValueSet = [getText(attr_value_gen_params[1]) for _ in range(attr_value_gen_params[0])]

    return tagSet, attrSet, attrValueSet

def generateXml(generatorArgs, vocabs=None):
    if vocabs is None:
        vocabs = generateVocabs(generatorArgs)
    (tagSet, attrSet, attrValueSet) = vocabs

    for k, v in generatorArgs.items():
        if v is None:
            del generatorArgs[k]

    node_count_range = generatorArgs.get("node_count_range", (1, 4))
    max_child_count = generatorArgs.get("max_child_count", 4)

    attr_count_range = generatorArgs.get("attr_count_range", (0, 5))
    text_len_range = generatorArgs.get("text_len_range", (-5, 7))
    tail_len_range = generatorArgs.get("tail_len_range", (-20, 10))

    root_node = None
    node_count = random.randrange(*node_count_range)
    permissibleParents = []

    for i in range(node_count):
        if permissibleParents:
            pickedParent = random.randrange(0, len(permissibleParents))
            parent = permissibleParents[pickedParent]
            if len(parent) >= max_child_count-1:
                del permissibleParents[pickedParent]
        else:
            parent = None

        # Build tag and attrs for the node.
        tag = random.choice(tagSet)
        attrCount = random.randrange(*attr_count_range)
        attrs = {}
        if attrCount > 0:
            for _ in range(attrCount):
                attrValue = random.choice(attrValueSet)
                attrs[random.choice(attrSet)] = attrValue

        # Instantiate node.
        node = ET.Element(tag, attrs)
        permissibleParents.append(node)

        # Build text and tail fields.
        node.text = getText(text_len_range)
        node.tail = getText(tail_len_range)

        # Mark parent.
        if parent is not None:
            parent.append(node)

        if root_node is None:
            root_node = node

    # Root node cannot have tail.
    root_node.tail = None

    return ET.ElementTree(root_node)
