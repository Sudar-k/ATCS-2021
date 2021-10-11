mountains = {
    'Mount Everest': '8848 meters',
    'K2': '8611 meters',
    'Lhotse' : '8516 meters',
    'Makalu' : '8485 meters',
    'Nanda Devi' : '7816 meters'
}
for name in mountains.keys():
    print(name)

print()

for height in mountains.values():
    print(height)

print()

for name, height in mountains.items():
    print(name, "is",height, "tall")