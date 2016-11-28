import improc
import number
import owner


# todo: simple UI and user-IO

def main(fname):
    card_img = improc.detect_card(fname)
    number_img = improc.detect_card_number(card_img)
    owner_img = improc.detect_card_owner(card_img)
    number.recognize_number(number_img)
    owner.recognize_owner(owner_img)

main('res/test2.jpg')
