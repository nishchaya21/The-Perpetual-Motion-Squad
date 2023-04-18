from colorthief import ColorThief
import webcolors
 
def det(img_pth):
    color_thief = ColorThief(img_pth)
    # get the dominant color
    dominant_color = color_thief.get_color(quality=1)
    def closest_colour(requested_colour):
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def get_colour_name(requested_colour):
        try:
            cn = actual_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            cn = closest_colour(requested_colour)
            actual_name = None
        return actual_name, cn

    requested_colour = dominant_color
    actual_name, cn = get_colour_name(requested_colour)
    if "black" in cn or "brown" in cn or "gray" in cn:
        return("Bad")
    elif "red" in cn or "orange" in cn or "coral" in cn or "tomato" in cn or "sienna" in cn:
        return("Moderate")
    elif "red" in cn or "yellow" in cn or "Lemon" in cn or "Gold" in cn  or "olive" in cn:
        return("Good")
    else:
        return("Bad")