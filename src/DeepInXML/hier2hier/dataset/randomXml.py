import random, string, re   
import xml.etree.ElementTree as ET

idStartVocab = string.ascii_letters + "_"
idVocab = string.digits + idStartVocab + "."
textVocab = re.sub("[.<>/]", "", string.printable)

def getId(len_range):
    return (
        random.choice(idStartVocab)
        + "".join([
            random.choice(idVocab) for _ in range(*len_range)
        ])
    )

def getText(len_range, vocab=None):
    textLen = random.randrange(*len_range)
    if textLen <= 0:
        return None

    if vocab is None:
        vocab = textVocab
    return "".join([random.choice(textVocab) for _ in range(textLen)])

def generateXml(generatorArgs):
    for k, v in generatorArgs.items():
        if v is None:
            del generatorArgs[k]

    node_count_range = generatorArgs.get("node_count_range", (1, 4))
    max_child_count = generatorArgs.get("max_child_count", 4)
    taglen_range = generatorArgs.get("taglen_range", (0, 5))

    attr_len_range = generatorArgs.get("attr_len_range", (0, 5))
    attr_value_len_range = generatorArgs.get("attr_value_len_range", (0, 20))
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
        tag = getId(taglen_range)
        attrCount = random.randrange(*attr_count_range)
        attrs = {}
        if attrCount > 0:
            for _ in range(attrCount):
                attrValue = getText(attr_value_len_range)
                attrs[getId(attr_len_range)] = attrValue

        # Instantiate node.
        node = ET.Element(tag, attrs)
        permissibleParents.append(node)

        # Build text and tail fields.
        node.text = getText(text_len_range)
        node.tail = getText(tail_len_range)

        # Mark parent.
        if parent:
            parent.append(node)

        if root_node is None:
            root_node = node

    return ET.ElementTree(root_node)