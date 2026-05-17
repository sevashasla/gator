python generate.py --name patch.room.livingroom.lounge \
  --prompts "a modern apartment living room with a sofa, coffee table, rug, shelves, and large windows" \
            "a furnished hotel lounge with armchairs, low tables, plants, carpet, and large windows" \
  --style "a photorealistic Matterport-style indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.kitchen.dining \
  --prompts "a residential kitchen with white cabinets, an island counter, bar stools, pendant lights, and tiled floor" \
            "a dining room with a long table, chairs, side cabinets, pendant lights, and tiled floor" \
  --style "a photorealistic wide angle RGB-D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.bedroom.office \
  --prompts "a tidy bedroom with a bed, nightstands, desk, chair, wardrobe, carpet, and soft daylight" \
            "a home office with a large desk, chair, bookshelves, storage cabinets, carpet, and soft daylight" \
  --style "a photorealistic Matterport-style indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.bathroom.laundry \
  --prompts "a clean bathroom with sink vanity, mirror, shower glass, white tiles, towels, and cabinets" \
            "a compact laundry room with washer dryer, utility sink, white cabinets, tiles, towels, and shelves" \
  --style "a photorealistic wide angle indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.hallway.gallery \
  --prompts "a long apartment hallway with doors, framed pictures, ceiling lights, wood floor, and white walls" \
            "a small art gallery corridor with framed pictures, benches, ceiling lights, wood floor, and white walls" \
  --style "a photorealistic Matterport-style indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.stairwell.lobby \
--prompts "a residential stairwell with wooden stairs, railings, landing, wall lights, and neutral walls" \
        "a small building lobby with stairs, railings, benches, wall lights, and neutral walls" \
--style "a photorealistic wide angle RGB-D scan rendering of" \
--views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.library.study \
  --prompts "a cozy library room with bookshelves, reading chairs, tables, lamps, carpet, and tall windows" \
            "a university study room with bookshelves, desks, chairs, lamps, carpet, and tall windows" \
  --style "a photorealistic indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.conference.classroom \
  --prompts "a conference room with a long table, office chairs, wall screen, whiteboard, carpet, and ceiling lights" \
            "a classroom with rows of desks, chairs, wall screen, whiteboard, carpet, and ceiling lights" \
  --style "a photorealistic Matterport-style indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.restaurant.cafe \
  --prompts "a small restaurant interior with dining tables, chairs, bar counter, pendant lights, plants, and tiled floor" \
            "a cozy cafe interior with small tables, chairs, service counter, pendant lights, plants, and tiled floor" \
  --style "a photorealistic wide angle indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.gym.playroom \
  --prompts "a compact indoor gym with exercise mats, benches, mirrors, storage shelves, rubber floor, and bright lights" \
            "a children's playroom with floor mats, benches, wall mirrors, storage shelves, rubber floor, and bright lights" \
  --style "a photorealistic RGB-D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.storage.closet \
  --prompts "a storage room with metal shelves, boxes, cabinets, concrete floor, fluorescent lights, and plain walls" \
            "a walk-in closet with shelves, boxes, cabinets, carpeted floor, ceiling lights, and plain walls" \
  --style "a photorealistic indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.museum.livingroom \
  --prompts "a museum room with display cases, benches, framed wall art, polished floor, spotlights, and neutral walls" \
            "a formal living room with glass cabinets, benches, framed wall art, polished floor, lamps, and neutral walls" \
  --style "a photorealistic Matterport-style indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.hotelroom.bedroom \
  --prompts "a hotel room with a bed, desk, armchair, curtains, lamps, carpet, and framed wall art" \
            "a residential bedroom with a bed, desk, armchair, curtains, lamps, carpet, and framed wall art" \
  --style "a photorealistic wide angle indoor 3D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0

python generate.py --name patch.room.openoffice.laboratory \
  --prompts "an open office with desks, chairs, monitors, shelves, ceiling lights, gray floor, and glass partitions" \
            "a research lab room with workbenches, stools, monitors, shelves, ceiling lights, gray floor, and glass partitions" \
  --style "a photorealistic RGB-D scan rendering of" \
  --views identity patch_permute --num_samples 10 --num_inference_steps 30 --guidance_scale 10.0
