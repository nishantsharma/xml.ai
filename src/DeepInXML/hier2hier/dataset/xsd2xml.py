#!/usr/bin/python2.7

import random, decimal, string
from argparse import ArgumentParser

import xmlschema
from xmlschema.validators import (
    XsdElement,
    XsdAnyElement,
    XsdComplexType,
    XsdAtomicBuiltin,
    XsdSimpleType,
    XsdList,
    XsdUnion
)

class GenXML:
    def __init__(self, xsd, elem, enable_choice=False):
        self.xsd = xmlschema.XMLSchema(xsd)
        self.elem = elem
        self.enable_choice = enable_choice
        self.root = True

    # shorten the namespace
    def short_ns(self, ns):
        for k, v in self.xsd.namespaces.items():
            if k == '':
                continue
            if v == ns:
                return k
        return ''

    # if name is using long namespace,
    # lets replace it with the short one
    def use_short_ns(self, name):
        if name[0] == '{':
            x = name.find('}')
            ns = name[1:x]
            return self.short_ns(ns) + ":" + name[x + 1:]
        return name


    # remove the namespace in name
    def remove_ns(self, name):
        if name[0] == '{':
            x = name.find('}')
            return name[x + 1:]
        return name

    # header of xml doc
    def print_header(self, fout):
        print("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>", file=fout)


    # put all defined namespaces as a string
    def ns_map_str(self):
        ns_all = ''
        for k, v in self.xsd.namespaces.items():
            if k == '':
                continue
            else:
                ns_all += 'xmlns:' + k + '=\"' + v + '\"' + ' '
        return ns_all


    # start a tag with name
    def start_tag(self, node, name):
        x = '<' + name
        if self.root:
            self.root = False
            x += ' ' + self.ns_map_str()

        for attr, attrValue in node.attributes.items():
            x += " " + attr + "=\"" + self.genval(attrValue.type.id) + "\""

        x += '>'
        return x


    # end a tag with name
    def end_tag(self, name):
        return '</' + name + '>'


    # make a sample data for primitive types
    def genval(self, name):
        name = self.remove_ns(name)

        # numeric types
        if   name == 'decimal':
            retval = decimal.Decimal(random.randrange(-1000, 1000))/100
        elif name == 'float':
            exponent = random.randint(-38, 38)
            retval = random.uniform(-1, 1) * pow(10, exponent)
        elif name == 'double':
            exponent = random.randint(sys.float_info.min_10_exp, sys.floatinfo.max_10_exp)
            retval = random.uniform(-1, 1) * pow(10, exponent)
        elif name in ['integer', 'long', 'int', 'short']:
            retval = random.randint(-32768, 32767)
        elif name == 'positiveInteger':
            retval = random.randint(0, 32767)
        elif name == 'negativeInteger':
            retval = random.randint(-32768, 0)
        elif name == 'nonPositiveInteger':
            retval = random.randint(-32768, -1)
        elif name == 'nonNegativeInteger':
            retval = random.randint(1, 32767)
        elif name == 'byte':
            retval = random.randint(-128, 127)
        elif name == ['unsignedLong', 'unsignedInt', 'unsignedShort']:
            retval = random.randint(0, 65535)
        elif name == 'unsignedByte':
            retval = random.randint(0, 255)
        # time/duration types
        elif name == 'dateTime':
            retval = '2004-04-12T13:20:00-05:00'
        elif name == 'date':
            retval = '2004-04-12'
        elif name == 'gYearMonth':
            retval = '2004-04'
        elif name == 'gYear':
            retval = '2004'
        elif name == 'duration':
            retval = 'P2Y6M5DT12H35M30S'
        elif name == 'dayTimeDuration':
            retval = 'P1DT2H'
        elif name == 'yearMonthDuration':
            retval = 'P2Y6M'
        elif name == 'gMonthDay':
            retval = '--04-12'
        elif name == 'gDay':
            retval = '---02'
        elif name == 'gMonth':
            retval = '--04'
        # string types
        elif name == 'string':
            min_char = 8
            max_char = 12
            allchar = string.ascii_letters + " ,.+-!:;?'\"@()[]{}^|" + string.digits
            retval = "".join([random.choice(allchar) for x in range(random.randint(min_char, max_char))])
        elif name == 'normalizedString':
            retval = 'The cure for boredom is curiosity.'
        elif name == 'token':
            retval = 'There is no cure for curiosity.'
        elif name == 'language':
            retval = 'en-US'
        elif name == 'NMTOKEN':
            retval = 'A_BCD'
        elif name == 'NMTOKENS':
            retval = 'ABCD 123'
        elif name == 'Name':
            retval = 'myElement'
        elif name == 'NCName':
            retval = '_my.Element'
        # magic types
        elif name == 'ID':
            retval = 'IdID'
        elif name == 'IDREFS':
            retval = 'IDrefs'
        elif name == 'ENTITY':
            retval = 'prod557'
        elif name == 'ENTITIES':
            retval = 'prod557 prod563'
        # oldball types
        elif name == 'QName':
            retval = 'pre:myElement'
        elif name == 'boolean':
            retval = 'true'
        elif name == 'hexBinary':
            retval = '0FB8'
        elif name == 'base64Binary':
            retval = '0fb8'
        elif name == 'anyURI':
            retval = 'http://miaozn.github.io/misc'
        elif name == 'notation':
            retval = 'asd'
        else:
            raise NotImplementedError("Unknown XML element name {0}.".format(name))

        retval = str(retval)
        return retval

    # print a group
    def group2xml(self, g, fout):
        model = str(g.model)
        model = self.remove_ns(model)
        nextg = g._group
        y = len(nextg)
        if y == 0:
            print('<!--empty-->', file=fout)
            return

        print('<!--START:[' + model + ']-->', file=fout)
        if self.enable_choice and model == 'choice':
            print('<!--next item is from a [choice] group with size=' + str(y) + '-->', file=fout)
        else:
            print('<!--next ' + str(y) + ' items are in a [' + model + '] group-->', file=fout)

        for ng in nextg:
            if isinstance(ng, XsdElement):
                self.node2xml(ng, fout)
            elif isinstance(ng, XsdAnyElement):
                self.node2xml(ng, fout)
            else:
                self.group2xml(ng, fout)

            if self.enable_choice and model == 'choice':
                break
        print('<!--END:[' + model + ']-->', file=fout)


    # print a node
    def node2xml(self, node, fout):
        if node.min_occurs == 0:
            print('<!--next 1 item is optional (minOcuurs = 0)-->', file=fout)
        if node.max_occurs >  1:
            print('<!--next 1 item is multiple (maxOccurs > 1)-->', file=fout)

        for _ in range(random.randint(node.min_occurs, node.max_occurs)):
            if isinstance(node, XsdAnyElement):
                print('<_ANY_/>', file=fout)
            elif isinstance(node.type, XsdComplexType):
                n = self.remove_ns(node.name)
                if node.type.is_simple():
                    print('<!--simple content-->', file=fout)
                    tp = str(node.type.content_type)
                    print(self.start_tag(node, n) + self.genval(tp) + self.end_tag(n), file=fout)
                else:
                    print('<!--complex content-->', file=fout)
                    print(self.start_tag(node, n), file=fout)
                    self.group2xml(node.type.content_type, fout)
                    print(self.end_tag(n), file=fout)
            elif isinstance(node.type, XsdAtomicBuiltin):
                n = self.remove_ns(node.name)
                tp = node.type.id
                print(self.start_tag(node, n) + self.genval(tp) + self.end_tag(n), file=fout)
            elif isinstance(node.type, XsdSimpleType):
                n = self.remove_ns(node.name)
                if isinstance(node.type, XsdList):
                    print('<!--simpletype: list-->', file=fout)
                    tp = str(node.type.item_type)
                    print(self.start_tag(node, n) + self.genval(tp) + self.end_tag(n), file=fout)
                elif isinstance(node.type, XsdUnion):
                    print('<!--simpletype: union.-->', file=fout)
                    print('<!--default: using the 1st type-->', file=fout)
                    tp = str(node.type.member_types[0].base_type)
                    print(self.start_tag(node, n) + self.genval(tp) + self.end_tag(n), file=fout)
                else:
                    tp = str(node.type.base_type)
                    print(self.start_tag(node, n) + self.genval(tp) + self.end_tag(n), file=fout)
            else:
                raise NotImplemented('ERROR: unknown type: ' + node.type)


    # setup and print everything
    def run(self, fout):
        self.print_header(fout)
        self.node2xml(self.xsd.elements[self.elem], fout)


##############

def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--schema", dest="xsdfile", required=True,
                        help="select the xsd used to generate xml")
    parser.add_argument("-o", "--output", dest="xmlfile", required=True,
                        help="select the XML used to write xml into")
    parser.add_argument("-e", "--element", dest="element", required=True,
                        help="select an element to dump xml")
    parser.add_argument("-c", "--choice",
                        action="store_true", dest="enable_choice", default=False,
                        help="enable the <choice> mode")
    args = parser.parse_args()

    generator = GenXML(args.xsdfile, args.element, args.enable_choice)
    with open(args.xmlfile, "w") as fout:
        generator.run(fout)



if __name__ == "__main__":
    main()



